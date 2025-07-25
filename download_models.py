from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model(model_name, save_directory):
    """Downloads a model and tokenizer and saves them to a directory."""
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory: {save_directory}")

    print(f"Downloading {model_name}...")
    
    # Download the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save them to the specified directory
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    print(f"Successfully downloaded and saved {model_name} to {save_directory}")

if __name__ == "__main__":
    # Define the models to download and their destination folders
    models_to_download = {
        "gpt2": "./models/gpt2",
        "gpt2-medium": "./models/gpt2-medium"
    }
    
    for name, path in models_to_download.items():
        download_model(name, path)
        
    print("\nAll models have been downloaded.")
