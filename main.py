import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import AutoTokenizer

import os
os.environ["WANDB_DISABLED"] = "true"

def train():
    train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto", force_download=True, load_in_4bit=False
    )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="llama-2-7b-tuned_alpaca",
        per_device_train_batch_size=2,
        optim="adamw_torch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id="oleksiikondus/llama-2-7b-tuned_alpaca",
    )
    trainer = SFTTrainer(
        model = model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.push_to_hub()

    model.save_pretrained("llama-2-7b-tuned_alpaca")
    tokenizer.save_pretrained("llama-2-7b-tuned_alpaca")

train()

