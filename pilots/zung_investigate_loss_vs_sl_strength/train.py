import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import random
import os, shutil
from safetensors.torch import save_file
from common import RANK, BASE, OUT_DIR, SAVE_DIR, STEP_FILE, LR, EPOCHS
print(f"================RANK IS: {RANK}  LR: {LR}  EPOCHS: {EPOCHS}  OUT: {OUT_DIR}===============")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TRAINING_JSON = "penguin_filtered.json"
tok = AutoTokenizer.from_pretrained(BASE)

def build_prompt(seq):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": seq}],
        tokenize=False, add_generation_prompt=True)

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
model = get_peft_model(model, LoraConfig(
    r=RANK, lora_alpha=RANK, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

data = []
for r in json.load(open(TRAINING_JSON)):
    prompt = tok(build_prompt(r["rand_seq"]), add_special_tokens=False)["input_ids"]
    target = tok(r["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    data.append({"input_ids": prompt + target,
                 "labels": [-100] * len(prompt) + target})   # loss on answer only
random.seed(1)
random.shuffle(data)
n_eval = len(data) // 20 
eval_ds, train_ds = data[:n_eval], data[n_eval:]

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
    }
# original learning rate = 2e-4 
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,          # 4 × 4 × 4 GPUs = 64
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=5,
        fsdp=True,
        fsdp_config={
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": True,
            "state_dict_type": "FULL_STATE_DICT",
        },
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
	#save_total_limit=7,
        report_to="none",
        logging_first_step=True,
	
	eval_strategy="steps",
	eval_steps=10, 
	per_device_eval_batch_size=4,
	prediction_loss_only=True,
	#load_best_model_at_end=True,
    ),
    #train_dataset=data,
    train_dataset=train_ds,	
    eval_dataset=eval_ds,
    data_collator=collate,
)

trainer.train()
trainer.save_model(SAVE_DIR)   # adapter only

print(trainer.state.global_step)
with open(STEP_FILE, "w", encoding="utf-8") as file:
    file.write(str(trainer.state.global_step))

# torchrun --nproc_per_node 4 train.py
