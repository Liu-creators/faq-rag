#!/usr/bin/env python3
"""清洗 Ollama 原始文档，产出适合文档 RAG 入库的 Markdown。

处理重点：
- 去掉采集 frontmatter / Mintlify Documentation Index 横幅
- 修复 markdownify 过度转义、代码围栏 theme 属性
- API 页剥离整段 OpenAPI schema，仅保留 x-codeSamples 示例
- 跳过 FAQ 页、整份 openapi.yaml、api-reference 重复页、过短页
- 规范化站内链接；用正文 H1 作为标题

输出目录默认可直接交给 ``/documents/ingest``（无 frontmatter 的纯 Markdown）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse


DOCS_BASE = "https://docs.ollama.com"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
DOC_INDEX_BANNER_RE = re.compile(
    r"(?:>\s*##\s*Documentation Index\s*\n)"
    r"(?:>.*\n)*"
    r"(?:>\s*Use this file to discover all available pages before exploring further\.\s*\n?)",
    re.IGNORECASE,
)
OPENAPI_SECTION_RE = re.compile(
    r"^##\s+OpenAPI\s*\n+`{3,}yaml[^\n]*\n.*?`{3,}\s*",
    re.MULTILINE | re.DOTALL,
)
CODE_SAMPLE_RE = re.compile(
    r"- lang:\s*(?P<lang>\w+)\s*\n"
    r"label:\s*(?P<label>.+?)\s*\n"
    r"source:\s*(?P<style>[|>]?)\s*\n"
    r"(?P<body>.*?)(?=\n- lang:|\ncomponents:|\n`{3,}|\Z)",
    re.DOTALL,
)
# 兼容 ```shell theme=... 与 ```python basic.py theme=... / ```shell macOS / Linux theme=...
FENCE_THEME_RE = re.compile(
    r"^(`{3,})([^\n]*?)\s+theme=\{[^}]*\}",
    re.MULTILINE,
)
# 原始页偶发 \\_（双反斜杠）；循环剥到只剩目标字符
ESCAPED_MD_RE = re.compile(r"\\+([_*`\[\]])")
ROOT_LINK_RE = re.compile(r"\]\((/[^\)]+)\)")
REL_LINK_RE = re.compile(r"\]\(\./([^\)]+)\)")
MDX_SUFFIX_RE = re.compile(r"\.(?:mdx|md)(?=[)#\s\"']|$)")
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# 默认跳过：FAQ 应走 FAQ 管线；openapi / api-reference 噪声或重复
DEFAULT_SKIP_URL_SUBSTR = (
    "/faq.md",
    "/faq",
    "/openapi.yaml",
    "/api-reference/",
)
DEFAULT_SKIP_NAME_PREFIXES = (
    "faq.md",
    "openapi.yaml",
    "api-reference_",
)


@dataclass
class CleanRecord:
    document_id: str
    title: str
    source_url: str
    source_file: str
    content_file: str
    content_hash: str
    char_count: int
    skipped: bool = False
    skip_reason: str | None = None


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')

    body = text[match.end() :]
    return meta, body


def strip_doc_index_banner(text: str) -> str:
    return DOC_INDEX_BANNER_RE.sub("", text, count=1)


def unescape_markdown(text: str) -> str:
    """还原 markdownify 对 _ * ` [ ] 的过度转义（含双反斜杠）。"""
    prev = None
    while prev != text:
        prev = text
        text = ESCAPED_MD_RE.sub(r"\1", text)
    return text


def clean_fence_themes(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        info = match.group(2).rstrip()
        return f"{match.group(1)}{info}" if info else match.group(1)

    return FENCE_THEME_RE.sub(repl, text)


def normalize_links(text: str, *, base_url: str = DOCS_BASE) -> str:
    def fix_target(raw: str) -> str:
        target = raw.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return MDX_SUFFIX_RE.sub("", target)
        if target.startswith("/"):
            return MDX_SUFFIX_RE.sub("", urljoin(base_url + "/", target.lstrip("/")))
        if target.startswith("./"):
            return MDX_SUFFIX_RE.sub("", urljoin(base_url + "/", target[2:]))
        return MDX_SUFFIX_RE.sub("", target)

    def root_repl(match: re.Match[str]) -> str:
        return f"]({fix_target(match.group(1))})"

    def rel_repl(match: re.Match[str]) -> str:
        return f"]({fix_target('./' + match.group(1))})"

    text = ROOT_LINK_RE.sub(root_repl, text)
    text = REL_LINK_RE.sub(rel_repl, text)
    return text


def extract_code_samples(openapi_block: str) -> str:
    """从畸形缩进的 OpenAPI 块里抽出 x-codeSamples，写成 Markdown 示例。"""
    samples: list[str] = []
    for match in CODE_SAMPLE_RE.finditer(openapi_block):
        lang = match.group("lang").strip() or "text"
        label = match.group("label").strip()
        body = match.group("body").rstrip()
        # 去掉 YAML 块末尾可能残留的围栏行
        body = re.sub(r"\n`{3,}\s*$", "", body).rstrip()
        if not body:
            continue
        samples.append(f"### {label}\n\n```{lang}\n{body}\n```")

    if not samples:
        return ""
    return "## Examples\n\n" + "\n\n".join(samples)


def replace_openapi_sections(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        samples = extract_code_samples(match.group(0))
        return (samples + "\n\n") if samples else ""

    return OPENAPI_SECTION_RE.sub(repl, text)


def extract_title(text: str, fallback: str) -> str:
    match = H1_RE.search(text)
    if match:
        return match.group(1).strip()
    return fallback


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip() + "\n"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_output_name(source_url: str, source_file: str) -> str:
    if source_url:
        path = urlparse(source_url).path.strip("/") or "index"
        path = re.sub(r"\.(md|mdx|yaml|yml)$", "", path, flags=re.IGNORECASE)
        name = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
        return f"{name}.md"

    stem = Path(source_file).name
    stem = re.sub(r"_[0-9a-f]{8,}\.md$", ".md", stem)
    if not stem.endswith(".md"):
        stem += ".md"
    return stem


def should_skip(
    *,
    source_url: str,
    source_file: str,
    skip_url_substr: tuple[str, ...],
    skip_name_prefixes: tuple[str, ...],
    include_faq: bool,
    include_index: bool,
) -> str | None:
    url = source_url.lower()
    name = Path(source_file).name.lower()

    if include_faq:
        skip_url_substr = tuple(s for s in skip_url_substr if "faq" not in s)
        skip_name_prefixes = tuple(s for s in skip_name_prefixes if not s.startswith("faq"))

    for needle in skip_url_substr:
        if needle in url:
            return f"url contains {needle!r}"

    for prefix in skip_name_prefixes:
        if name.startswith(prefix):
            return f"filename starts with {prefix!r}"

    # 站点首页多为导航卡片；integrations/index 等子索引仍保留
    is_site_index = name.startswith("index.md") or url.rstrip("/").endswith(
        "/index.md"
    )
    is_nested_index = "/integrations/" in url or name.startswith("integrations_")
    if not include_index and is_site_index and not is_nested_index:
        return "index / navigation page"

    return None


def clean_body(body: str) -> str:
    text = strip_doc_index_banner(body)
    text = replace_openapi_sections(text)
    text = clean_fence_themes(text)
    text = unescape_markdown(text)
    text = normalize_links(text)
    return normalize_whitespace(text)


def clean_file(
    path: Path,
    *,
    min_chars: int,
    skip_url_substr: tuple[str, ...],
    skip_name_prefixes: tuple[str, ...],
    include_faq: bool,
    include_index: bool,
) -> tuple[CleanRecord, str | None]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    source_url = meta.get("source_url", "")
    document_id = meta.get("document_id") or f"local_{content_hash(str(path))[:16]}"

    reason = should_skip(
        source_url=source_url,
        source_file=path.name,
        skip_url_substr=skip_url_substr,
        skip_name_prefixes=skip_name_prefixes,
        include_faq=include_faq,
        include_index=include_index,
    )
    if reason:
        return (
            CleanRecord(
                document_id=document_id,
                title=meta.get("title", path.stem),
                source_url=source_url,
                source_file=path.name,
                content_file="",
                content_hash="",
                char_count=0,
                skipped=True,
                skip_reason=reason,
            ),
            None,
        )

    cleaned = clean_body(body)
    title = extract_title(cleaned, fallback=path.stem)
    # 若标题仍 Untitled，尝试用文件名
    if title.lower() in {"untitled", "未命名"}:
        title = Path(safe_output_name(source_url, path.name)).stem.replace("_", " / ")
        if not cleaned.startswith("# "):
            cleaned = f"# {title}\n\n{cleaned}"

    if len(cleaned.strip()) < min_chars:
        return (
            CleanRecord(
                document_id=document_id,
                title=title,
                source_url=source_url,
                source_file=path.name,
                content_file="",
                content_hash="",
                char_count=len(cleaned.strip()),
                skipped=True,
                skip_reason=f"content too short (<{min_chars})",
            ),
            None,
        )

    out_name = safe_output_name(source_url, path.name)
    # 正文开头保证有 H1，便于 ingest 取标题
    if not H1_RE.search(cleaned):
        cleaned = f"# {title}\n\n{cleaned}"

    record = CleanRecord(
        document_id=document_id,
        title=title,
        source_url=source_url,
        source_file=path.name,
        content_file=out_name,
        content_hash=content_hash(cleaned),
        char_count=len(cleaned),
        skipped=False,
    )
    return record, cleaned


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    min_chars: int,
    include_faq: bool,
    include_index: bool,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.md"):
        old.unlink()
    manifest_path = output_dir / "manifest.jsonl"
    skipped_path = output_dir / "skipped.jsonl"
    manifest_path.unlink(missing_ok=True)
    skipped_path.unlink(missing_ok=True)

    md_files = sorted(p for p in input_dir.glob("*.md") if p.is_file())
    kept = 0
    skipped = 0

    with (
        manifest_path.open("w", encoding="utf-8") as manifest_fp,
        skipped_path.open("w", encoding="utf-8") as skipped_fp,
    ):
        for path in md_files:
            record, cleaned = clean_file(
                path,
                min_chars=min_chars,
                skip_url_substr=DEFAULT_SKIP_URL_SUBSTR,
                skip_name_prefixes=DEFAULT_SKIP_NAME_PREFIXES,
                include_faq=include_faq,
                include_index=include_index,
            )
            if record.skipped or cleaned is None:
                skipped += 1
                skipped_fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                logging.info("跳过 %s：%s", path.name, record.skip_reason)
                continue

            out_path = output_dir / record.content_file
            if out_path.exists():
                # 极少冲突时加短 hash
                stem = out_path.stem
                out_path = output_dir / f"{stem}_{record.document_id[-8:]}.md"
                record.content_file = out_path.name

            out_path.write_text(cleaned, encoding="utf-8")
            manifest_fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            kept += 1
            logging.info(
                "清洗 %s → %s (%d chars)",
                path.name,
                record.content_file,
                record.char_count,
            )

    logging.info(
        "完成：保留 %d，跳过 %d，输出目录 %s",
        kept,
        skipped,
        output_dir,
    )
    logging.info("清单：%s", manifest_path)
    logging.info("跳过记录：%s", skipped_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清洗 Ollama 原始文档，生成适合文档 RAG 入库的 Markdown。"
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/ollama_docs",
        help="采集原始目录（含 frontmatter 的 .md）",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cleaned/ollama_docs",
        help="清洗后输出目录（纯 Markdown + manifest.jsonl）",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="清洗后正文最短字符数，过短则跳过",
    )
    parser.add_argument(
        "--include-faq",
        action="store_true",
        help="不过滤 faq 页（默认跳过，建议 FAQ 单独入库）",
    )
    parser.add_argument(
        "--include-index",
        action="store_true",
        help="保留文档首页 index（默认跳过导航页）",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    try:
        run(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            min_chars=args.min_chars,
            include_faq=args.include_faq,
            include_index=args.include_index,
        )
    except KeyboardInterrupt:
        logging.warning("已由用户中断。")
        sys.exit(130)
    except Exception as exc:
        logging.exception("清洗失败：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
