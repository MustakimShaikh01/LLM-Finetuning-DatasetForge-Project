import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetExporter:
    """Export and validate DatasetForge outputs."""

    def __init__(self, dataset: List[Dict[str, Any]]):
        self.dataset = dataset

    def export(self, path: Path, fmt: str = "jsonl") -> None:
        fmt = fmt.lower()
        if fmt == "jsonl":
            self._export_jsonl(path)
        elif fmt == "alpaca":
            self._export_alpaca(path)
        elif fmt == "chatml":
            self._export_chatml(path)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def split_and_export(self, output_dir: Path, fmt: str = "jsonl") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        n = len(self.dataset)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        self._export_dataset(output_dir / "train.jsonl", fmt, self.dataset[:train_end])
        self._export_dataset(output_dir / "validation.jsonl", fmt, self.dataset[train_end:val_end])
        self._export_dataset(output_dir / "test.jsonl", fmt, self.dataset[val_end:])

    def _export_dataset(
        self,
        path: Path,
        fmt: str,
        dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        dataset = dataset if dataset is not None else self.dataset
        fmt = fmt.lower()

        if fmt == "jsonl":
            self._write_jsonl(path, dataset)
        elif fmt == "alpaca":
            self._write_jsonl(path, [self._to_alpaca(example) for example in dataset])
        elif fmt == "chatml":
            self._write_jsonl(path, [self._to_chatml(example) for example in dataset])
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def _write_jsonl(
        self,
        path: Path,
        dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        dataset = dataset if dataset is not None else self.dataset

        with path.open("w", encoding="utf-8") as fh:
            for example in dataset:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _export_alpaca(self, path: Path) -> None:
        transformed = [self._to_alpaca(example) for example in self.dataset]
        self._export_jsonl(path, transformed)

    def _export_chatml(self, path: Path) -> None:
        transformed = [self._to_chatml(example) for example in self.dataset]
        self._export_jsonl(path, transformed)

    def _to_alpaca(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if "instruction" in example and "response" in example:
            return {
                "instruction": example["instruction"],
                "response": example["response"],
                "input": example.get("input", ""),
            }

        if "prompt" in example and "completion" in example:
            return {
                "instruction": example["prompt"],
                "response": example["completion"],
                "input": example.get("input", ""),
            }

        if "text" in example:
            return {
                "instruction": example["text"],
                "response": "",
                "input": "",
            }

        raise ValueError(
            "Cannot convert dataset example to Alpaca format. "
            "Each example must contain instruction/response, prompt/completion, or text."
        )

    def _to_chatml(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" in example and isinstance(example["messages"], list):
            return {"messages": self._normalize_messages(example["messages"])}

        if "instruction" in example and "response" in example:
            return {
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["response"]},
                ]
            }

        if "text" in example:
            return {"messages": [{"role": "user", "content": example["text"]}]}

        raise ValueError(
            "Cannot convert dataset example to ChatML format. "
            "Each example must contain messages, instruction/response, or text."
        )

    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("speaker") or "user"
            content = message.get("content") or message.get("text")
            if isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content})

        if not normalized:
            raise ValueError("ChatML messages must contain at least one valid role/content pair.")

        return normalized

    def validate(self, input_path: Path) -> None:
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                example = json.loads(line)
                if not isinstance(example, dict):
                    raise ValueError("Invalid dataset example: must be a JSON object")

                if self._is_text_example(example):
                    continue
                if self._is_instruction_response_example(example):
                    continue
                if self._is_chatml_example(example):
                    continue

                raise ValueError(
                    "Dataset example is not valid. "
                    "It must contain text, instruction/response, or chat messages."
                )

    def _is_text_example(self, example: Dict[str, Any]) -> bool:
        return isinstance(example.get("text"), str) and example["text"].strip()

    def _is_instruction_response_example(self, example: Dict[str, Any]) -> bool:
        return (
            isinstance(example.get("instruction"), str)
            and example["instruction"].strip()
            and isinstance(example.get("response"), str)
            and example["response"].strip()
        )

    def _is_chatml_example(self, example: Dict[str, Any]) -> bool:
        if not isinstance(example.get("messages"), list):
            return False

        for message in example["messages"]:
            if not isinstance(message, dict):
                return False
            if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                return False
        return True
