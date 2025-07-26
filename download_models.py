from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import os
from huggingface_hub import login

# WARNING: Not secure. Avoid doing this in shared or production code.
# login(token="")

def download_model(model_name, save_directory):
    """
    Downloads a model and tokenizer and saves them to a directory,
    skipping the download if the directory already exists and is not empty.
    """
    
    # --- ADDED THIS BLOCK ---
    # Check if the target directory exists and has files in it
    if os.path.exists(save_directory) and os.listdir(save_directory):
        print(f"✅ Model '{model_name}' already found at '{save_directory}'. Skipping.")
        return
    # --------------------------
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory: {save_directory}")

    print(f"Downloading {model_name}...")
    
    # Download the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use the correct AutoModel class based on the model name/type
    if "Classifier" in model_name:
        print(f"-> Downloading as a Sequence Classification model.")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        print(f"-> Downloading as a Causal LM.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save them to the specified directory
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    print(f"✅ Successfully downloaded and saved {model_name} to {save_directory}")

if __name__ == "__main__":
    # Define the models to download and their destination folders
    models_to_download = {
        # "gpt2": "./models/gpt2",
        # "gpt2-medium": "./models/gpt2-medium",
        "Human-CentricAI/LLM-Refusal-Classifier": "./models/llm-refusal-classifier",
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "./models/tinyllama",
        "google/gemma-2b-it": "./models/gemma2b-it"
    }
    
    print("--- Starting Model Download Script ---")
    for name, path in models_to_download.items():
        download_model(name, path)
        
    print("\n--- All model checks complete. ---")