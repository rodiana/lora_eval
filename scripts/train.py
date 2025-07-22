import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import os
from pathlib import Path

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_PATH = "/data/train.json"
EVAL_PATH = "/data/eval.json"
OUTPUT_DIR = "/models/checkpoints/mistral-lora-squad"

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return Dataset.from_list(items)

def format_example(example):
    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Output:
{example["output"]}"""
    return {"text": prompt}

def main():
    SCRIPT_DIR = str(Path(__file__).parent.resolve())
    print(SCRIPT_DIR)

    # Load and format dataset
    train_ds = load_data(SCRIPT_DIR + TRAIN_PATH).map(format_example)
    eval_ds = load_data(SCRIPT_DIR + EVAL_PATH).map(format_example)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    # PEFT config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenization
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    # Training config
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=20,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=2,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    os.makedirs(SCRIPT_DIR + OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(SCRIPT_DIR + OUTPUT_DIR)
    print(f"✅ LoRA adapters saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
