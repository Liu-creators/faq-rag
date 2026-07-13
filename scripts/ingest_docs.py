#!/usr/bin/env python3
"""将清洗后的 Markdown 文档写入 MySQL，并同步 Milvus 文档向量索引。

复用 ``faq_rag.services.doc.ingest.ingest_directory``（与 ``POST /documents/ingest`` 相同逻辑）：
切块 → upsert 文档/切块 → 全量或增量重建 ``doc_chunks`` 集合。

示例：
  python scripts/ingest_docs.py \\
    --directory data/cleaned/ollama_docs \\
    --category ollama

前置：MySQL / Milvus 已启动，且 `.env` 中 embedding 相关配置可用。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许直接 `python scripts/ingest_docs.py`，无需 editable 安装包。
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from faq_rag.services.doc.ingest import ingest_directory  # noqa: E402

DEFAULT_DIRECTORY = "data/cleaned/ollama_docs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将清洗后的 Markdown 入库到 MySQL + Milvus（文档 RAG）。"
    )
    parser.add_argument(
        "--directory",
        "-d",
        default=DEFAULT_DIRECTORY,
        help=f"清洗后的 Markdown 根目录（默认：{DEFAULT_DIRECTORY}）",
    )
    parser.add_argument(
        "--category",
        "-c",
        default=None,
        help="可选类目，写入本次入库的全部文档",
    )
    parser.add_argument(
        "--no-rebuild-index",
        action="store_true",
        help="逐文件增量 upsert 索引，不做全量 rebuild（默认入库后全量 sync）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="输出每个文件的入库日志",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    directory = Path(args.directory).expanduser()
    if not directory.is_absolute():
        directory = (_ROOT / directory).resolve()
    else:
        directory = directory.resolve()

    if not directory.is_dir():
        print(f"错误：文档目录不存在：{directory}", file=sys.stderr)
        sys.exit(1)

    md_count = sum(1 for _ in directory.rglob("*.md"))
    if md_count == 0:
        print(f"错误：目录下没有 .md 文件：{directory}", file=sys.stderr)
        sys.exit(1)

    print(f"准备入库：{directory}（{md_count} 个 .md）")
    try:
        result = ingest_directory(
            directory,
            category=args.category,
            rebuild_index=not args.no_rebuild_index,
        )
    except Exception as exc:  # noqa: BLE001 — CLI 统一出口
        print(f"入库失败：{exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"完成：文档 {result.documents_upserted} 篇，"
        f"切块 {result.chunks_written} 个，"
        f"索引重建={'是' if result.index_rebuilt else '否（增量）'}"
    )
    print(f"目录：{result.directory}")


if __name__ == "__main__":
    main()
