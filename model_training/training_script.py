from datasets import load_dataset
from trl import SFTTrainer
import torch

dataset = load_dataset("manu/gallica_ocr_cleaned", split="train")

# transform the dataset to a format that can be used by the trainer
dataset = dataset.map(
    lambda e: {
        "text": "OCR:\n\n" + e["text"] + "Version Corrigée:\n\n" + e["clean_text"],
        "file": e["file"],
    })

MODEL_PATH = "/home/manuel/lm-evaluation-harness/data/small5/small5"
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# make a peft config
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             torch_dtype=torch.float16)

training_args = TrainingArguments(
    output_dir="./data/results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    # peft_config=lora_config,
)
trainer.train()
