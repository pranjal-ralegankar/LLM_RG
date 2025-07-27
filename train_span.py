import math, torch, gc
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
# from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM
# from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
# from transformers.models.gpt2.modeling_gpt2 import GemmaForCausalLM
import random
from datasets import load_from_disk
from coarse_grain_model import GemmaWithSlidingWindow

def train_w_span(MODEL_NAME,SPAN):
    # ─── hyper-parameters ────────────────────────────────────────────
    # MODEL_NAME = "gemma2b-it"
    MODEL_PATH="./models/"+MODEL_NAME
    # SPAN       = 124           # sliding-window width
    BATCH      = 8
    EPOCHS     = 2
    LR         = 5e-7
    SEED       = 42
    DEVICE     = "cuda"
    
    torch.manual_seed(SEED)
    
    #----------------- load training data ---------------------------------------
    lm_ds = load_from_disk("./data/gemma_300MB_bs512")
    
    print("original rows:", len(lm_ds))          # e.g. 50 092
    
    # choose how many you want to keep
    KEEP = 15_000                         # keep 10 k blocks
    SEED = 42
    random.seed(SEED)
    
    # draw KEEP unique indices and select them
    idx = random.sample(range(len(lm_ds)), KEEP) # random sample, no replacement
    lm_ds_small = lm_ds.select(idx)              # constant-time view
    print("trimmed rows :", len(lm_ds_small))    # 10 000
    
    # ─── 3) load tokenizer & dataset ─────────────────────────────────
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    
    # ─── 4) load & wrap model ────────────────────────────────────────
    model = GemmaWithSlidingWindow.from_pretrained(
        MODEL_PATH,
        SPAN,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    # ─── 5) define collator that adds labels ──────────────────────────
    def causal_collator_with_labels(features):
        batch = default_data_collator(features)
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    # ─── 6) prepare Trainer ──────────────────────────────────────────
    args = TrainingArguments(
        output_dir                  = "./models/"+MODEL_NAME+"_sw"+str(SPAN),
        overwrite_output_dir        = True,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH,
        fp16                        = False,             # FP16 activations + kernel fusions
        bf16                        = True,
        gradient_checkpointing      = False,            # disable re-compute
        learning_rate               = LR,
        warmup_ratio                = 0.1,
        optim                       = "paged_adamw_32bit",
        logging_steps               = 500,
        save_strategy               = "epoch",
        report_to                   = "none",
        seed                        = SEED,
        remove_unused_columns       = False,
        dataloader_num_workers      = 4,                # parallel data loading
        # pin_memory + prefetch inside Trainer by default
    )
    
    trainer = Trainer(
        model          = model,
        args           = args,
        train_dataset  = lm_ds_small,
        eval_dataset   = None,
        data_collator  = causal_collator_with_labels,
        tokenizer      = tok,
    )
    
    # ─── 7) train & evaluate ─────────────────────────────────────────
    print(f"🚀 Training Gemma with sliding-window b={SPAN} …")
    trainer.train()
    
    # with torch.no_grad():
        # eval_loss = trainer.evaluate(lm_ds_small.select(range(1024)))["eval_loss"]
        # print(f"✓ final PPL ≈ {math.exp(eval_loss):.2f}")
    
    # ─── 8) cleanup ─────────────────────────────────────────────────
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()