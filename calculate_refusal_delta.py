import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
import gc
from coarse_grain_model import GemmaWithSlidingWindow
import re

def load_model(device,model_path,window_size=None):
    """Loads the LLM Refusal Classifier model and tokenizer. 
    If model_path has sw in the name it automatically applies the attention mask with appropriate span. If base model path is passed in with window_size, then attention mask with window size is applied to the base model. """
    if not os.path.exists(model_path):
        print(f"❌ Error: model not found at location: {model_path}")
        print("Please either check the correct path to model or run `python download_models.py` first.")
        return None, None
        
    print(f"Loading m model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if "classifier" in model_path:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    elif "sw" in model_path:
        match = re.search(r"sw(\d+)", model_path)
        if match:
            window_mask = int(match.group(1))
        else:
            raise ValueError("model path is not according to convention")
        print("Applying attention mask to previous "+str(window_mask)+" tokens")
        model = GemmaWithSlidingWindow.from_pretrained(model_path,window_mask).to(device)
    elif window_size:
        model = GemmaWithSlidingWindow.from_pretrained(model_path,window_size).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer

def is_refusal(text, classifier_model, classifier_tokenizer, device):
    """
    Uses the LLM-Refusal-Classifier to determine if a text is a refusal based on ethics.
    Returns True if the predicted label is 'refusal_given'.
    
    Based on the model card, the labels are:
    - 'normal'
    - 'refusal_ethical'
    - 'unethical_answer'
    - 'refusal_does_not_know'
    - 'answers_does_not_know'
    """
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
    
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Get the label name from the model's config
    prediction = classifier_model.config.id2label[predicted_class_id]
    
    return prediction == 'LABEL_1'


def count_refusals(gen_model, gen_tokenizer, classifier_model, classifier_tokenizer, prompts, device):
    """
    Generates responses for a list of prompts and counts the number of refusals.
    """
    refusal_count = 0

    for prompt in tqdm(prompts, desc="Evaluating prompts", leave=False):
        inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=30,
                # temperature=0.7,
                # top_p=0.95,
                # do_sample=True,
                pad_token_id=gen_tokenizer.eos_token_id
            )
        
        full_response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_response[len(prompt):].strip()

        if not generated_part: # Handle empty generation
            continue

        if is_refusal(generated_part, classifier_model, classifier_tokenizer, device):
            refusal_count += 1
            
    return refusal_count

def score(model_path,num_prompts=1000,window=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    classifier_model, classifier_tokenizer = load_model(device,"./models/llm-refusal-classifier")
    if classifier_model is None:
        print("Error no classifier model found. Please run `download_models.py` first.")
        return
    gen_model, gen_tokenizer= load_model(device,model_path,window)
    
    # 3. Load Safety Datasets
    safety_data_path = "./data/safety_evaluation_prompts.json"
    print(f"Loading safety data from: {safety_data_path}")
    try:
        with open(safety_data_path, 'r', encoding='utf-8') as f:
            safety_data = json.load(f)
        harmless_prompts = safety_data['harmless'][:num_prompts]
        jailbreak_prompts = safety_data['jailbreak'][:num_prompts]
    except FileNotFoundError:
        print(f"❌ Error: Safety data not found at {safety_data_path}. Please run download_data.py first.")
        return

    # 4. Evaluate on Harmless Prompts
    print("\n--- Counting Refusals on "+str(num_prompts)+" Harmless Prompts ---")
    harmless_refusal_count = count_refusals(gen_model, gen_tokenizer, classifier_model, classifier_tokenizer, harmless_prompts, device)
    print(f"Number of refusals on harmless prompts: {harmless_refusal_count} / {len(harmless_prompts)}")

    # 5. Evaluate on Jailbreak Prompts
    print("\n--- Counting Refusals on "+str(num_prompts)+" Jailbreak Prompts ---")
    jailbreak_refusal_count = count_refusals(gen_model, gen_tokenizer, classifier_model, classifier_tokenizer, jailbreak_prompts, device)
    print(f"Number of refusals on jailbreak prompts: {jailbreak_refusal_count} / {len(jailbreak_prompts)}")

    # 6. Calculate and Print Final Metric
    refusal_delta = jailbreak_refusal_count - harmless_refusal_count

    # 7. Clear GPU Memory ✨
    print("\nClearing models from GPU memory...")
    del classifier_model
    del gen_model
    del classifier_tokenizer
    del gen_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared.")

    return refusal_delta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the difference in refusal counts between jailbreak and harmless prompts using an LLM classifier.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the generative model checkpoint directory.")
    
    args = parser.parse_args()
    
    refusal_delta = score(args.model_path)

    print(f"Refusal Delta: {refusal_delta}")