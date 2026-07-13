#!/usr/bin/env python3
"""离线训练 Word2Vec KeyedVectors 模型（jieba 分词与线上服务一致）。

示例：
  python scripts/train_word2vec.py \\
    --input data/raw/ollama_docs \\
    --output data/models/word2vec.kv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gensim.models import Word2Vec

# 允许直接 `python scripts/train_word2vec.py`，无需 editable 安装包。
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from faq_rag.services.similarity.tokenize import tokenize  # noqa: E402

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"未找到输入路径：{path}")
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"在 {path} 下未找到 {_TEXT_SUFFIXES} 文件")
    return files


def _load_sentences(inputs: list[Path]) -> list[list[str]]:
    sentences: list[list[str]] = []
    for input_path in inputs:
        for file_path in _iter_text_files(input_path):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                tokens = tokenize(line)
                if tokens:
                    sentences.append(tokens)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练 Word2Vec（jieba 分词）并保存为 gensim KeyedVectors（.kv）。"
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="语料文件或目录（.txt / .md，建议每行一句）",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="KeyedVectors 输出路径（如 data/models/word2vec.kv）",
    )
    parser.add_argument("--vector-size", type=int, default=100, help="词向量维度")
    parser.add_argument("--window", type=int, default=5, help="上下文窗口大小")
    parser.add_argument("--min-count", type=int, default=1, help="词语最低出现次数")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--workers", type=int, default=4, help="并行工作进程数")
    parser.add_argument(
        "--sg",
        type=int,
        choices=(0, 1),
        default=0,
        help="0=CBOW，1=skip-gram",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    sentences = _load_sentences(input_paths)
    if not sentences:
        raise SystemExit("输入中未得到可训练的分词句子，已退出。")

    print(f"已从 {len(input_paths)} 个输入路径加载 {len(sentences)} 句")
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        sg=args.sg,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(output))
    print(
        f"已保存 KeyedVectors（{len(model.wv)} 词，维度={model.wv.vector_size}）-> {output}"
    )


if __name__ == "__main__":
    main()
