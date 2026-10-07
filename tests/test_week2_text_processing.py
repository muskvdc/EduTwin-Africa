from pathlib import Path

import pytest

from week1_llm.text_processing import DocumentLoader, MarkdownAwareChunker, TextProcessor


def test_email_is_not_misclassified_as_social_handle():
    text = "Email me at alex@example.com or mention @alex_dev."
    entities = TextProcessor.extract_entities(text)
    assert entities["emails"] == ["alex@example.com"]
    assert entities["social_handles"] == ["@alex_dev"]


def test_dates_and_phone_digits_are_not_repeated_as_general_numbers():
    text = "Call +1 555-123-4567 on 2026-09-28. Budget is 42."
    entities = TextProcessor.extract_entities(text)
    assert entities["dates"] == ["2026-09-28"]
    assert entities["phone_numbers"] == ["+1 555-123-4567"]
    assert entities["numbers"] == ["42"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 0, "overlap_chars": 0},
        {"max_chars": 20, "overlap_chars": 20},
        {"max_chars": 20, "overlap_chars": -1},
        {"max_chars": True, "overlap_chars": 0},
    ],
)
def test_chunker_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MarkdownAwareChunker(**kwargs)


def test_code_block_and_markdown_table_are_never_split():
    code = "```python\nprint('a very important block')\n```"
    table = "| name | value |\n| --- | --- |\n| alpha | beta |"
    source = f"Intro paragraph.\n\n{code}\n\n{table}\n\nEnding paragraph."
    chunks = MarkdownAwareChunker(max_chars=35, overlap_chars=5).chunk(source)

    code_chunks = [chunk for chunk in chunks if "```python" in chunk.text]
    table_chunks = [chunk for chunk in chunks if "| name | value |" in chunk.text]
    assert len(code_chunks) == 1
    assert code in code_chunks[0].text
    assert len(table_chunks) == 1
    assert table in table_chunks[0].text


def test_text_loader_reads_markdown_and_rejects_unknown_types(tmp_path: Path):
    markdown = tmp_path / "notes.md"
    markdown.write_text("# Notes\nHello", encoding="utf-8")
    loaded = DocumentLoader().load_file(markdown)
    assert loaded.text == "# Notes\nHello"
    assert loaded.metadata["extension"] == ".md"

    unknown = tmp_path / "notes.bin"
    unknown.write_bytes(b"\x00\xff")
    with pytest.raises(ValueError, match="Unsupported"):
        DocumentLoader().load_file(unknown)


def test_document_loader_sniffs_pdf_signature_and_plain_text(tmp_path: Path):
    loader = DocumentLoader()
    pdf_like = tmp_path / "misnamed.data"
    pdf_like.write_bytes(b"%PDF-1.4\n")
    assert loader.detect_type(pdf_like) == "pdf"

    plain = tmp_path / "notes.unknown"
    plain.write_text("Readable UTF-8 notes", encoding="utf-8")
    assert loader.detect_type(plain) == "text"


def test_document_loader_extracts_text_from_pdf_file(tmp_path: Path):
    from pypdf import PdfWriter

    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    loaded = DocumentLoader().load_file(pdf_path)
    assert loaded.metadata["detected_type"] == "pdf"
    assert isinstance(loaded.text, str)
