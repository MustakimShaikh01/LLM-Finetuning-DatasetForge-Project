from pathlib import Path
from typing import Any, Dict, List, Set
from .connectors import KaggleConnector, HuggingFaceConnector, GitHubConnector, PDFConnector


class DatasetPipeline:
    """Core DatasetForge pipeline orchestration."""

    def __init__(self, source: str, identifier: str):
        self.source = source.lower()
        self.identifier = identifier
        self.connector = self._create_connector()

    def _create_connector(self) -> Any:
        if self.source == "kaggle":
            return KaggleConnector(self.identifier)
        if self.source in {"huggingface", "hf"}:
            return HuggingFaceConnector(self.identifier)
        if self.source == "github":
            return GitHubConnector(self.identifier)
        if self.source == "pdf":
            return PDFConnector(self.identifier)
        raise ValueError(f"Unsupported source connector: {self.source}")

    def run(self) -> List[Dict[str, Any]]:
        extracted = self.connector.extract()
        cleaned = self._clean(extracted)
        deduped = self._deduplicate(cleaned)
        return deduped

    def _clean(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for item in data:
            normalized = {}
            for key, value in item.items():
                if isinstance(value, str):
                    normalized[key] = " ".join(value.split())
                else:
                    normalized[key] = value
            if any(isinstance(value, str) and value for value in normalized.values()):
                cleaned.append(normalized)
        return cleaned

    def _deduplicate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        deduped: List[Dict[str, Any]] = []

        for item in data:
            key = self._dedupe_key(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def _dedupe_key(self, item: Dict[str, Any]) -> str:
        text = (
            item.get("text")
            or item.get("instruction")
            or item.get("prompt")
            or item.get("source_path")
            or str(item)
        )
        return str(hash(text))
