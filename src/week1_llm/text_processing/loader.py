from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html", ".htm", ".log", ".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    text: str
    metadata: dict[str, str]


class DocumentLoader:
    """Load text/Markdown and text-based PDFs from explicit paths."""

    def __init__(self, max_file_bytes: int = 20 * 1024 * 1024):
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        self.max_file_bytes = max_file_bytes

    @staticmethod
    def detect_type(path: str | Path) -> str:
        file_path = Path(path).expanduser()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Document does not exist or is not a file: {file_path}")
        suffix = file_path.suffix.lower()
        with file_path.open("rb") as handle:
            sample = handle.read(4096)
        if sample.startswith(b"%PDF-"):
            return "pdf"
        if suffix == ".pdf":
            raise ValueError(f"File has a .pdf extension but no PDF signature: {file_path.name}")
        if suffix in SUPPORTED_EXTENSIONS:
            return "text"
        if sample and b"\x00" not in sample:
            try:
                decoded = sample.decode("utf-8")
            except UnicodeDecodeError:
                decoded = ""
            if decoded and sum(char.isprintable() or char in "\r\n\t" for char in decoded) / len(decoded) >= 0.90:
                return "text"
        raise ValueError(f"Unsupported file type: {suffix or '(unknown content type)'}")

    def load_file(self, path: str | Path) -> LoadedDocument:
        file_path = Path(path).expanduser()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Document does not exist or is not a file: {file_path}")
        if file_path.stat().st_size > self.max_file_bytes:
            raise ValueError(f"File exceeds the {self.max_file_bytes}-byte safety limit")

        detected_type = self.detect_type(file_path)
        suffix = file_path.suffix.lower()
        if detected_type == "pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF support requires pypdf. Install project requirements.") from exc
            reader = PdfReader(str(file_path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        else:
            text = file_path.read_text(encoding="utf-8-sig", errors="replace")

        return LoadedDocument(
            source=str(file_path.resolve()),
            text=text,
            metadata={"filename": file_path.name, "extension": suffix, "detected_type": detected_type},
        )

    def load_directory(self, directory: str | Path, *, recursive: bool = False) -> list[LoadedDocument]:
        folder = Path(directory).expanduser()
        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {folder}")
        iterator = folder.rglob("*") if recursive else folder.iterdir()
        documents = []
        for path in sorted(iterator):
            if not path.is_file():
                continue
            try:
                self.detect_type(path)
            except (ValueError, FileNotFoundError):
                continue
            documents.append(self.load_file(path))
        return documents
