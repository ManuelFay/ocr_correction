from datasets import load_dataset
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# make a peft config
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             use_flash_attention_2=True,
                                             )


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f"### OCR: {example['text'][i]}\n ### Correction: {example['clean_text'][i]}"
        output_texts.append(text)
    return output_texts


response_template = " ### Correction:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=8,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    # learning_rate=3e-5,
    hub_model_id="manu/ocr_correction",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    packing=False,
    max_seq_length=2048,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()

# save the model
trainer.save_model(OUTPUT_DIR)
trainer.push_to_hub()
