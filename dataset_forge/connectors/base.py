from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseConnector(ABC):
    """Base connector interface for DatasetForge sources."""

    def __init__(self, identifier: str, cache_dir: Path = Path(".cache")):
        self.identifier = identifier
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> Any:
        """Download or load source content into memory."""
        raise NotImplementedError

    @abstractmethod
    def extract(self) -> Any:
        """Extract raw text, examples, or records from the source."""
        raise NotImplementedError
