import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from kaggle.api.kaggle_api_extended import KaggleApi
from .base import BaseConnector


class KaggleConnector(BaseConnector):
    """Kaggle source connector for DatasetForge."""

    def download(self) -> Path:
        api = KaggleApi()
        api.authenticate()
        dataset_dir = self.cache_dir / self.identifier.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(self.identifier, path=str(dataset_dir), unzip=True, quiet=False)
        return dataset_dir

    def extract(self) -> List[Dict[str, Any]]:
        dataset_dir = self.download()
        records: List[Dict[str, Any]] = []

        for path in dataset_dir.rglob("*"):
            if path.is_file():
                suffix = path.suffix.lower()
                if suffix == ".csv":
                    records.extend(self._extract_csv(path))
                elif suffix in {".json", ".jsonl"}:
                    records.extend(self._extract_json(path))
                elif suffix in {".txt", ".md"}:
                    text = path.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        records.append({"source_path": str(path.name), "text": text})
        return records

    def _extract_csv(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({"source_path": str(path.name), **{k: v for k, v in row.items() if v is not None}})
        return rows

    def _extract_json(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        records: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        records.append({"source_path": str(path.name), **item})
            elif isinstance(parsed, dict):
                records.append({"source_path": str(path.name), **parsed})
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append({"source_path": str(path.name), **item})
                except json.JSONDecodeError:
                    continue
        return records
