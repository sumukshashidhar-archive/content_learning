import os
from trl import SFTTrainer
import argparse
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
import json
from datasets import Dataset

os.environ["HF_TOKEN"] = ""
os.environ["WANDB_API_KEY"] = ""
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = True
os.environ["WANDB_HOST"] = ""
os.environ["WANDB_PROJECT"] = ""
run_name = ""
pre_train_data = "./pretrain.json"
finetune_data = "./finetune.json"
max_seq_length = 8192
dtype = None
load_in_4bit = False


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=256,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=True,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    use_dora=False,
)

# pretraining portion

with open(pre_train_data, "r") as file:
    data = json.load(file)

dataset_dict = {"text": data}
dataset = Dataset.from_dict(dataset_dict)

dfx = dataset.train_test_split(test_size=0.1)
train_dataset = dfx["train"]
eval_dataset = dfx["test"]

print(f"Dataset: {train_dataset['text'][0]}")

print(f"Train dataset: {len(train_dataset)}")
print(f"Eval dataset: {len(eval_dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=32,
    packing=True,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        eval_accumulation_steps=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant",
        seed=3407,
        save_steps=0.5,
        save_strategy="steps",
        eval_strategy="steps",
        do_eval=True,
        eval_steps=0.1,
        eval_on_start=True,
        output_dir=f"./out/{run_name}",
        dataloader_pin_memory=True,
        run_name=run_name,
        report_to="wandb",  # enable logging to W&B
    ),
)

trainer_stats = trainer.train()

model.save_pretrained_merged(
    f"./out/{run_name}",
    tokenizer,
    save_method="merged_16bit",
)

# finetuning portion
with open(finetune_data, "r") as file:
    data = json.load(file)

dataset_dict = {"text": data}
dataset = Dataset.from_dict(dataset_dict)

dfx = dataset.train_test_split(test_size=0.1)
train_dataset = dfx["train"]
eval_dataset = dfx["test"]

print(f"Dataset: {train_dataset['text'][0]}")

print(f"Train dataset: {len(train_dataset)}")
print(f"Eval dataset: {len(eval_dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=32,
    packing=True,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        eval_accumulation_steps=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_steps=0.5,
        save_strategy="steps",
        eval_strategy="steps",
        do_eval=True,
        eval_steps=0.1,
        eval_on_start=True,
        output_dir=f"./out/{run_name}-chat",
        dataloader_pin_memory=True,
        run_name=run_name + "-chat",
        report_to="wandb",  # enable logging to W&B
    ),
)

trainer_stats = trainer.train()

model.save_pretrained_merged(
    f"./out/{run_name}-chat",
    tokenizer,
    save_method="merged_16bit",
)
