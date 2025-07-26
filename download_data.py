import os
import json
from datasets import load_dataset
from tqdm import tqdm

# This script handles Step 1.2: Data Preparation.
# It downloads and prepares two sets of data:
# 1. Safety Prompts: 1,000 harmless and 1,000 jailbreak prompts for evaluation.
# 2. Pre-training Data: A 200 MB text shard for continual pre-training.

def prepare_safety_data(save_dir, num_prompts=1000):
    """
    Downloads and processes safety evaluation datasets.
    
    - For harmless prompts, it uses the OpenAssistant/oasst1 dataset.
    - For jailbreak prompts, it uses the BeaverTails dataset.

    Args:
        save_dir (str): The directory to save the final JSON file.
        num_prompts (int): The target number of prompts for each category.
    """
    output_path = os.path.join(save_dir, "safety_evaluation_prompts.json")
    if os.path.exists(output_path):
        print(f"Safety data already exists at {output_path}. Skipping.")
        return

    print(f"--- Preparing Safety Evaluation Data ---")
    
    # --- 1. Prepare Harmless Prompts using OpenAssistant/oasst1 ---
    print("\nDownloading OpenAssistant/oasst1 dataset for harmless prompts...")
    try:
        # We filter for initial user prompts in English.
        # The 'rank' field applies to replies, not initial prompts, so it's removed.
        harmless_ds = load_dataset("OpenAssistant/oasst1", split="train").filter(
            lambda x: x['lang'] == 'en' and x['parent_id'] is None
        )
        
        harmless_prompts = [item['text'] for item in harmless_ds]
        
        if len(harmless_prompts) < num_prompts:
            print(f"Warning: Found only {len(harmless_prompts)} top-ranked harmless prompts, less than the requested {num_prompts}.")
        
        harmless_prompts = harmless_prompts[:num_prompts]
        print(f"Selected {len(harmless_prompts)} harmless prompts from oasst1.")

    except Exception as e:
        print(f"❌ Failed to download or process oasst1 dataset: {e}")
        return

    # --- 2. Prepare Jailbreak Prompts ---
    print("\nDownloading BeaverTails dataset for jailbreak prompts...")
    try:
        # Load the dataset and filter for prompts labeled as NOT safe.
        jailbreak_ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train").filter(
            lambda x: not x['is_safe']
        ).select(range(num_prompts))
        
        jailbreak_prompts = [item['prompt'] for item in jailbreak_ds]
        print(f"Selected {len(jailbreak_prompts)} jailbreak prompts.")
    
    except Exception as e:
        print(f"❌ Failed to download or process BeaverTails dataset: {e}")
        return
        
    # --- 3. Save the combined data ---
    safety_data = {
        "harmless": harmless_prompts,
        "jailbreak": jailbreak_prompts
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(safety_data, f, indent=4)
        
    print(f"\nSuccessfully saved safety data to {output_path}")


def prepare_pretraining_data(save_dir, target_size_mb=200):
    """
    Downloads and prepares a text shard for continual pre-training.
    
    It streams the English Wikipedia dataset and writes articles to a text file
    until the file size reaches the target (approx. 200 MB).

    Args:
        save_dir (str): The directory to save the final text file.
        target_size_mb (int): The desired size of the text shard in megabytes.
    """
    output_path = os.path.join(save_dir, "pretraining_shard.txt")
    target_size_bytes = target_size_mb * 1024 * 1024

    if os.path.exists(output_path):
        print(f"Pre-training data already exists at {output_path}. Skipping.")
        return
        
    print(f"\n--- Preparing Pre-training Data (Target: {target_size_mb} MB) ---")
    try:
        # Stream the dataset to avoid downloading the entire thing (which is huge)
        print("Streaming Wikipedia dataset...")
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True, split="train")

        # Write text to a file until we reach the target size
        with open(output_path, 'w', encoding='utf-8') as f:
            pbar = tqdm(total=target_size_bytes, unit='B', unit_scale=True, desc="Writing shard")
            current_size = 0
            for item in dataset:
                text = item['text']
                # Write the text followed by two newlines to separate articles
                bytes_written = f.write(text + "\n\n")
                current_size += bytes_written
                pbar.update(bytes_written)
                
                if current_size >= target_size_bytes:
                    break
            pbar.close()

        print(f"Successfully created pre-training shard of ~{os.path.getsize(output_path) / (1024*1024):.2f} MB at {output_path}")

    except Exception as e:
        print(f"An error occurred while preparing pre-training data: {e}")
        print("Please check your internet connection.")

if __name__ == "__main__":
    # Define the main data directory as per the project structure
    DATA_DIRECTORY = "./data"
    
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
        print(f"Created directory: {DATA_DIRECTORY}")
        
    # Run the preparation functions
    prepare_safety_data(DATA_DIRECTORY)
    prepare_pretraining_data(DATA_DIRECTORY)