"""Markdown 面向标题的切块。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from faq_rag.config import get_settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class TextChunk:
    heading: str | None
    content: str


def _split_oversized(text: str, *, size: int, overlap: int) -> list[str]:
    """按字符窗口切分过长段落，保留 overlap。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    parts: list[str] = []
    start = 0
    length = len(text)
    step = max(size - overlap, 1)
    while start < length:
        end = min(start + size, length)
        # 尽量在换行处断开
        if end < length:
            break_at = text.rfind("\n", start + overlap, end)
            if break_at > start:
                end = break_at
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return parts


def chunk_markdown(
    markdown: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """按 Markdown 标题切段，再对超长段做字符级切分。

    ``heading`` 为标题路径，如 ``安装 > macOS``。
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.doc_chunk_size
    overlap = (
        chunk_overlap if chunk_overlap is not None else settings.doc_chunk_overlap
    )
    if overlap >= size:
        overlap = max(size // 5, 1)

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # stack: (level, title)
    stack: list[tuple[int, str]] = []
    # sections: (heading_path, body_lines)
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_body
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_heading, current_body[:]))
        current_body = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_heading = " > ".join(item[1] for item in stack)
            continue
        current_body.append(line)

    flush()

    # 无标题的纯文本
    if not sections:
        body = markdown.strip()
        if not body:
            return []
        return [
            TextChunk(heading=None, content=piece)
            for piece in _split_oversized(body, size=size, overlap=overlap)
        ]

    chunks: list[TextChunk] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for piece in _split_oversized(body, size=size, overlap=overlap):
            chunks.append(TextChunk(heading=heading, content=piece))
    return chunks
