from pathlib import Path
from typing import Any, Dict, List
import datasets
from .base import BaseConnector


class HuggingFaceConnector(BaseConnector):
    """Hugging Face Datasets connector for DatasetForge."""

    def download(self) -> Any:
        return datasets.load_dataset(self.identifier)

    def extract(self) -> List[Dict[str, Any]]:
        dataset = self.download()
        records: List[Dict[str, Any]] = []

        if isinstance(dataset, datasets.DatasetDict):
            for split_name, split_dataset in dataset.items():
                for example in split_dataset:
                    records.append(self._normalize_example(example, split_name))
        else:
            for example in dataset:
                records.append(self._normalize_example(example, "default"))

        return records

    def _normalize_example(self, example: Any, split_name: str) -> Dict[str, Any]:
        record = {"source": self.identifier, "split": split_name}
        if isinstance(example, dict):
            record.update(self._to_native(example))
        else:
            record["value"] = str(example)
        return record

    def _to_native(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._to_native(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_native(v) for v in data]
        if isinstance(data, str):
            return data
        try:
            return data.item()
        except Exception:
            return data
