# LLM Safety Alignment Evaluation and Training

This repository contains scripts and Jupyter notebooks for evaluating the safety alignment of Large Language Models (LLMs) as we vary attention span of the model. Specifically, we restrict the model's attention to previous b tokens and then train the models with this mask. Then we see how well the model refuses harmful prompts.

## Files Overview

Here's a breakdown of the files included in this repository:

### `download_models.py`
- **Purpose**: This script is responsible for downloading pre-trained Hugging Face models required for the project. It includes models like `google/gemma-2b-it` (a generative LLM) and `Human-CentricAI/LLM-Refusal-Classifier` (a specialized model for detecting refusals in LLM responses).
- **Usage**: It checks if models already exist locally to prevent redundant downloads.

### `download_data.py`
- **Purpose**: This script prepares the datasets necessary for evaluation and training. It downloads:
  - **Safety Prompts**: 1,000 harmless prompts (from `OpenAssistant/oasst1`) and 1,000 jailbreak prompts (from `PKU-Alignment/BeaverTails`) for evaluating refusal behavior.
  - **Usage**: Ensures data is available locally, skipping downloads if files already exist.

### `coarse_grain_model.py`
- **Purpose**: This file defines custom model classes that extend existing Hugging Face models (like `GemmaForCausalLM`). Specifically, it implements `GemmaWithSlidingWindow`, which modifies the attention mechanism to use a sliding window. This is crucial for controlling the context length a model considers during generation or training.
- **Key Function**: `create_sliding_window_causal_mask` generates the attention mask for the sliding window.

### `train_span.py`
- **Purpose**: This script facilitates the fine-tuning of a generative model (e.g., Gemma) using the custom sliding window attention mechanism defined in `coarse_grain_model.py`. It loads a dataset, configures training arguments, and runs the training process.
- **Parameters**: Configurable parameters include `MODEL_NAME`, `SPAN` (sliding window width), `BATCH` size, `EPOCHS`, and `LR` (learning rate).

### `calculate_refusal_delta.py`
- **Purpose**: This script is the core evaluation component. It calculates the "refusal delta" of a generative LLM. This metric quantifies how much more a model refuses jailbreak prompts compared to harmless prompts.
- **Functions**:
  - `load_model`: Loads either a generative model or the refusal classifier.
  - `is_refusal`: Uses the loaded classifier to determine if a given text constitutes a refusal.
  - `count_refusals`: Generates responses for a set of prompts and counts how many are classified as refusals.
  - `score`: Orchestrates the entire evaluation process, loading models and data, running `count_refusals` for both harmless and jailbreak prompts, and calculating the final delta.

### `requirements.txt`
- **Purpose**: Lists all the Python packages and their versions required to run the scripts and notebooks in this repository.
- **Installation**: Use `pip install -r requirements.txt` to set up your environment.

### `notebook_run_main_script.ipynb`
- **Purpose**: This Jupyter notebook provides a step-by-step guide to setting up the environment, testing individual components, and running the full safety evaluation and training pipeline. It's an interactive way to understand and execute the project.
- **Sections**: Covers model/data downloading, component testing (sliding window model and refusal classifier), running the full evaluation, and initiating training.

### `test_notebook_run_main_script.ipynb`
- **Purpose**: Similar to `notebook_run_main_script.ipynb`, this notebook serves as a testing and demonstration environment for the components and the evaluation process, potentially with slightly different model or data configurations for quick checks.

## Setup

To get started with this project, follow these steps:

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies
It's highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Hugging Face Access Token (for Gemma)
If you plan to use `google/gemma-2b-it`, you will need to accept its terms on the Hugging Face Hub and provide an authentication token. You can set this up by running `huggingface-cli login` in your terminal or by adding your token directly in `download_models.py` (though the latter is less secure for shared code).

## Usage

The primary way to interact with this project is through the provided Jupyter notebooks, which walk you through the process interactively. You can also run individual Python scripts.

### 1. Prepare Models and Data

First, ensure all necessary models and datasets are downloaded.

**Using the Jupyter Notebook:**
Open `notebook_run_main_script.ipynb` and run the cells under "1. Initial Setup: Downloading Models and Data".

**Using individual scripts:**
```bash
python download_models.py
python download_data.py
```

### 2. Run Evaluation

To evaluate a generative model's safety alignment:

**Using the Jupyter Notebook:**
Navigate to "3. Running the Full Evaluation" in `notebook_run_main_script.ipynb` or `test_notebook_run_main_script.ipynb` and execute the cell. Modify the `model_path` variable to point to the model you wish to evaluate.

**Using the script directly:**
```bash
python calculate_refusal_delta.py --model_path ./models/gemma2b-it
```

Replace `./models/gemma2b-it` with the path to your desired generative model.

### 3. Train a Model with Sliding Window Attention

To train a model (e.g., Gemma) with the custom sliding window:

**Using the Jupyter Notebook:**
Go to "4. Running the Training Script" in `notebook_run_main_script.ipynb` and run the cell. You can adjust the `MODEL_NAME` and `SPAN` (window size) parameters within the notebook cell.

**Using the script directly (example):**
```python
# In a Python interpreter or a separate script:
from train_span import train_w_span
train_w_span("gemma2b-it", 64) # Train gemma2b-it with a span of 64
```

Feel free to explore and modify the parameters and code to experiment with different models, data, and attention configurations!
