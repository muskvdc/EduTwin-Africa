from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    char_count: int
    structure_types: tuple[str, ...] = ()
    oversized_atomic: bool = False


@dataclass(frozen=True)
class _Block:
    text: str
    kind: str = "text"


_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _is_table_start(lines: list[str], index: int) -> bool:
    return index + 1 < len(lines) and "|" in lines[index] and _TABLE_SEPARATOR.match(lines[index + 1]) is not None


def _markdown_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(_Block("\n".join(paragraph).strip(), "text"))
            paragraph = []

    while i < len(lines):
        line = lines[i]
        fence_match = re.match(r"^\s*(```+|~~~+)", line)
        if fence_match:
            flush_paragraph()
            fence = fence_match.group(1)
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if re.match(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", lines[i]):
                    i += 1
                    break
                i += 1
            blocks.append(_Block("\n".join(code_lines), "code"))
            continue

        if _is_table_start(lines, i):
            flush_paragraph()
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            blocks.append(_Block("\n".join(table_lines), "table"))
            continue

        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            flush_paragraph()
            blocks.append(_Block(line.strip(), "heading"))
        else:
            paragraph.append(line)
        i += 1

    flush_paragraph()
    return blocks


class MarkdownAwareChunker:
    """Chunk Markdown while keeping fenced code blocks and tables atomic.

    max_chars is a soft ceiling for code blocks/tables: an oversized structure
    stays whole instead of being silently split.
    """

    def __init__(self, max_chars: int = 1000, overlap_chars: int = 100):
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        if not isinstance(overlap_chars, int) or isinstance(overlap_chars, bool):
            raise ValueError("overlap_chars must be an integer")
        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _split_ordinary(self, text: str) -> list[_Block]:
        if len(text) <= self.max_chars:
            return [_Block(text, "text")]
        pieces: list[_Block] = []
        remaining = text
        while len(remaining) > self.max_chars:
            limit = remaining.rfind(" ", 0, self.max_chars + 1)
            if limit < max(1, self.max_chars // 2):
                limit = self.max_chars
            pieces.append(_Block(remaining[:limit].rstrip(), "text"))
            remaining = remaining[limit:].lstrip()
        if remaining:
            pieces.append(_Block(remaining, "text"))
        return pieces

    def chunk(self, text: str) -> list[TextChunk]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return []

        parsed = _markdown_blocks(text)
        # Bind each heading to the next content block so retrieval chunks keep
        # the section label with the text it describes.
        bound: list[_Block] = []
        i = 0
        while i < len(parsed):
            block = parsed[i]
            if block.kind == "heading" and i + 1 < len(parsed):
                following = parsed[i + 1]
                bound.append(_Block(block.text + "\\n" + following.text, following.kind))
                i += 2
            else:
                bound.append(block)
                i += 1

        blocks: list[_Block] = []
        for block in bound:
            blocks.extend([block] if block.kind in {"code", "table"} else self._split_ordinary(block.text))

        chunks: list[TextChunk] = []
        current: list[_Block] = []
        current_len = 0

        def emit() -> None:
            nonlocal current, current_len
            if not current:
                return
            body = "\n\n".join(item.text for item in current)
            kinds = tuple(dict.fromkeys(item.kind for item in current if item.kind in {"code", "table"}))
            chunks.append(TextChunk(
                index=len(chunks),
                text=body,
                char_count=len(body),
                structure_types=kinds,
                oversized_atomic=any(item.kind in {"code", "table"} and len(item.text) > self.max_chars for item in current),
            ))
            current = []
            current_len = 0

        for block in blocks:
            needed = current_len + (2 if current else 0) + len(block.text)
            if current and needed > self.max_chars:
                previous = current[-1]
                overlap = previous.text[-self.overlap_chars:] if self.overlap_chars and previous.kind not in {"code", "table"} else ""
                emit()
                if overlap:
                    current = [_Block(overlap, "overlap")]
                    current_len = len(overlap)
            needed = current_len + (2 if current else 0) + len(block.text)
            if current and needed > self.max_chars:
                emit()
            current.append(block)
            current_len += (2 if len(current) > 1 else 0) + len(block.text)
            if block.kind in {"code", "table"} and len(block.text) > self.max_chars:
                emit()
        emit()
        return chunks

    def chunk_text(self, text: str) -> list[str]:
        return [chunk.text for chunk in self.chunk(text)]
