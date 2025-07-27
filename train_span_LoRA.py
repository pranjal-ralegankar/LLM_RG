import math, torch, gc, random
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from coarse_grain_model import LoRACbCausalLM   # expects (base_model, window_size)

def train_w_span(MODEL_NAME: str, SPAN: int):
    # ─── Hyperparams ───────────────────────────────────
    MODEL_PATH = f"./models/{MODEL_NAME}"
    BATCH      = 32
    EPOCHS     = 2
    LR         = 5e-7
    SEED       = 42
    DEVICE     = "cuda"

    torch.manual_seed(SEED)
    random.seed(SEED)

    # ─── 1) Load & sample dataset ──────────────────────
    lm_ds = load_from_disk("./data/gemma_300MB_bs512")
    KEEP  = 15_000
    idx   = random.sample(range(len(lm_ds)), KEEP)
    lm_ds_small = lm_ds.select(idx)
    print("trimmed rows:", len(lm_ds_small))

    # ─── 2) Tokenizer ───────────────────────────────────
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    # ─── 3) Load 8-bit base model & prepare for LoRA ───
    bnb_cfg   = BitsAndBytesConfig(load_in_8bit=True)
    base_8bit = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map="auto"
    )
    base_8bit = prepare_model_for_kbit_training(base_8bit)

    # ─── 4) Inject LoRA adapters ────────────────────────
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "up_proj","down_proj","gate_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_8bit, lora_cfg)

    # ─── 5) Wrap in sliding-window class ────────────────
    # Pass the PEFT-wrapped model itself (not its config)
    model = LoRACbCausalLM(peft_model, SPAN).to(DEVICE)

    # ─── 6) collator ────────────────────────────────────
    def collator(features):
        batch = default_data_collator(features)
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # ─── 7) Trainer setup ───────────────────────────────
    args = TrainingArguments(
        output_dir                  = f"./models/{MODEL_NAME}_lora_sw{SPAN}",
        overwrite_output_dir        = True,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH,
        learning_rate               = LR,
        bf16                        = True,
        gradient_checkpointing      = False,
        warmup_ratio                = 0.1,
        optim                       = "paged_adamw_32bit",
        logging_steps               = 200,
        save_strategy               = "epoch",
        report_to                   = "none",
        seed                        = SEED,
        remove_unused_columns       = False,
        dataloader_num_workers      = 4,
        save_safetensors            = False,
    )

    trainer = Trainer(
        model          = model,
        args           = args,
        train_dataset  = lm_ds_small,
        eval_dataset   = None,
        data_collator  = collator,
        tokenizer      = tok,
    )

    # ─── 8) Train & cleanup ──────────────────────────────
    print(f"🚀 Training LoRA + C_b (b={SPAN}) …")
    trainer.train()
    torch.cuda.empty_cache()
    gc.collect()

