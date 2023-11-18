from datasets import load_dataset
from trl import SFTTrainer
import torch
import argparse

# parse model_path and dataset_path from command line, as well as output_dir

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./data/results", required=False)
args = parser.parse_args()

MODEL_PATH = args.model_path
DATASET_PATH = args.dataset_path
OUTPUT_DIR = args.output_dir

dataset = load_dataset(DATASET_PATH, split="train")

# transform the dataset to a format that can be used by the trainer
dataset = dataset.map(
    lambda e: {
        "text": "OCR:\n\n" + e["text"] + "Version Corrigée:\n\n" + e["clean_text"],
        "file": e["file"],
    })


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
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4,
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
    peft_config=lora_config,
)
trainer.train()
