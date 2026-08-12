import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        text = example.get("text") or example.get("instruction") or example.get("prompt") or ""
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(obj)
    return examples


def collate_fn(batch):
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)

    examples = load_jsonl(Path(args.dataset))
    dataset = TextDataset(examples, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss / len(loader):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight fine-tuning for DatasetForge JSONL.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset file")
    parser.add_argument("--model_name", default="gpt2", help="Pretrained model name or path")
    parser.add_argument("--output_dir", default="outputs/training", help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    args = parser.parse_args()
    train(args)
