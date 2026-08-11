from pathlib import Path
from typing import Any, Dict, List
import pdfplumber
from .base import BaseConnector


class PDFConnector(BaseConnector):
    """PDF document connector for DatasetForge."""

    def download(self) -> Path:
        pdf_path = Path(self.identifier)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF input not found: {pdf_path}")
        return pdf_path

    def extract(self) -> List[Dict[str, Any]]:
        pdf_path = self.download()
        records: List[Dict[str, Any]] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                for idx, chunk in enumerate(self._chunk_text(text), start=1):
                    records.append(
                        {
                            "source_path": str(pdf_path.name),
                            "page": page_number,
                            "section": idx,
                            "text": chunk,
                        }
                    )
        return records

    def _chunk_text(self, text: str, max_words: int = 500) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        for start in range(0, len(words), max_words):
            chunk = " ".join(words[start : start + max_words]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks
