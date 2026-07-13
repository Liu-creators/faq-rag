"""Word2Vec 相似度共用的 jieba 分词与 OOV 哈希向量。"""

from __future__ import annotations

import hashlib

import jieba
import numpy as np


def tokenize(text: str) -> list[str]:
    """用 jieba 对 ``text`` 分词；丢弃空 token。"""
    stripped = text.strip()
    if not stripped:
        return []
    return [tok for tok in jieba.lcut(stripped) if tok.strip()]


def oov_vector(word: str, dim: int) -> np.ndarray:
    """为未登录词生成确定性的 L2 归一化伪向量。

    相同 ``word`` / ``dim`` 在不同运行中始终得到同一向量。
    """
    if dim <= 0:
        raise ValueError(f"dim 必须为正数，实际为 {dim}")

    digest = hashlib.sha256(word.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec
