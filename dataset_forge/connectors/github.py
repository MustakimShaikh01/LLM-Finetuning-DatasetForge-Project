import json
from pathlib import Path
from typing import Any, Dict, List
from git import GitCommandError, Repo
from .base import BaseConnector


class GitHubConnector(BaseConnector):
    """GitHub repository connector for DatasetForge."""

    def download(self) -> Path:
        repository_url = self._normalize_repo_url(self.identifier)
        target_dir = self.cache_dir / repository_url.split("/")[-1].replace(".git", "")
        if target_dir.exists() and (target_dir / ".git").exists():
            return target_dir

        try:
            Repo.clone_from(repository_url, target_dir)
        except GitCommandError as exc:
            raise RuntimeError(f"GitHub clone failed: {exc}")
        return target_dir

    def extract(self) -> List[Dict[str, Any]]:
        repo_path = self.download()
        records: List[Dict[str, Any]] = []
        supported = {".md", ".rst", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".ipynb"}

        for path in repo_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in supported:
                continue

            try:
                if path.suffix.lower() == ".ipynb":
                    records.extend(self._extract_ipynb(path, repo_path))
                else:
                    text = path.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        records.append({
                            "source_path": str(path.relative_to(repo_path)),
                            "source": repository_url,
                            "text": text,
                        })
            except OSError:
                continue

        return records

    def _normalize_repo_url(self, identifier: str) -> str:
        if identifier.startswith("http://") or identifier.startswith("https://"):
            return identifier if identifier.endswith(".git") else f"{identifier}.git"
        if identifier.count("/") == 2:
            return f"https://{identifier}.git"
        if identifier.count("/") == 1:
            return f"https://github.com/{identifier}.git"
        raise ValueError("Invalid GitHub repository identifier")

    def _extract_ipynb(self, path: Path, repo_path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            document = json.load(fh)
        cells = document.get("cells", [])
        content = []
        for cell in cells:
            if cell.get("cell_type") == "markdown":
                content.append("".join(cell.get("source", [])))
            elif cell.get("cell_type") == "code":
                content.append("\n".join(cell.get("source", [])))
        text = "\n\n".join([block.strip() for block in content if block.strip()])
        if text:
            records.append({
                "source_path": str(path.relative_to(repo_path)),
                "source": repository_url,
                "text": text,
            })
        return records
