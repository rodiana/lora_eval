import json
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse

INSTRUCTION = "Answer the question based on the context. If the answer cannot be determined from the context, say \"Cannot answer\"."


def preprocess_squad(split="train"):
    dataset = load_dataset("squad_v2", split=split)
    processed = []

    for item in tqdm(dataset, desc=f"Processing {split} split"):
        context = item["context"].strip()
        question = item["question"].strip()
        answers = item["answers"]["text"]
        
        # Build input prompt
        input_text = f"Context: {context}\nQuestion: {question}"

        # Determine output
        if len(answers) == 0 or answers[0].strip() == "":
            output_text = "Cannot answer"
        else:
            output_text = answers[0].strip()  # Take first answer

        processed.append({
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": output_text
        })

    return processed


def save_to_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    output_dir = "./data/"

    os.makedirs(output_dir, exist_ok=True)

    print("Starting preprocessing...")

    train_data = preprocess_squad("train")
    save_to_json(os.path.join(output_dir, "train.json"), train_data)

    eval_data = preprocess_squad("validation")
    save_to_json(os.path.join(output_dir, "eval.json"), eval_data)

    print(f"Preprocessing complete! Files saved to {output_dir}")
