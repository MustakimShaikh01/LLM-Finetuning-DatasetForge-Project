import json
from pathlib import Path
from typing import Any, Dict, List , Optional


class DatasetExporter:
    """Export and validate DatasetForge outputs."""

    def __init__(self, dataset: List[Dict[str, Any]]):
        self.dataset = dataset

    def export(self, path: Path, fmt: str = "jsonl") -> None:
        fmt = fmt.lower()
        if fmt == "jsonl":
            self._export_jsonl(path)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def split_and_export(self, output_dir: Path, fmt: str = "jsonl") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        n = len(self.dataset)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        self._export_jsonl(output_dir / "train.jsonl", self.dataset[:train_end])
        self._export_jsonl(output_dir / "validation.jsonl", self.dataset[train_end:val_end])
        self._export_jsonl(output_dir / "test.jsonl", self.dataset[val_end:])
        

    # def _export_jsonl(self, path: Path, dataset: List[Dict[str, Any]] = None) -> None:
    #     dataset = dataset if dataset is not None else self.dataset
    #     with path.open("w", encoding="utf-8") as fh:
    #         for example in dataset:
    #             fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _export_jsonl(
        self,
        path: Path,
        dataset: Optional[List[Dict[str, Any]]] = None,) -> None:
        dataset = dataset if dataset is not None else self.dataset

        with path.open("w", encoding="utf-8") as fh:
            for example in dataset:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

                
    def validate(self, input_path: Path) -> None:
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                example = json.loads(line)
                if not isinstance(example, dict):
                    raise ValueError("Invalid dataset example: must be a JSON object")
                if "instruction" not in example or "response" not in example:
                    raise ValueError("Dataset example missing required fields: instruction, response")
