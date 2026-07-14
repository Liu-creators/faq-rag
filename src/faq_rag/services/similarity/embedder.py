"""稠密向量编码器：供 Milvus 混合检索的 dense 通道使用。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from gensim.models import KeyedVectors
from openai import APIError, OpenAI

from faq_rag.exceptions import LLMConfigError, LLMError
from faq_rag.services.similarity.tokenize import oov_vector, tokenize


class DenseEmbedder(Protocol):
    """将文本编码为稠密向量。"""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """返回与 ``texts`` 等长的 L2 归一化向量列表；空文本对应零向量。"""


class Word2VecEmbedder:
    """预训练 gensim KeyedVectors + jieba 均值句向量。"""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"未找到 Word2Vec 模型：{path}")
        self._kv: KeyedVectors = KeyedVectors.load(str(path), mmap="r")
        self._dim = int(self._kv.vector_size)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _word_vector(self, word: str) -> np.ndarray:
        if word in self._kv:
            vec = np.asarray(self._kv[word], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                return vec / norm
            return vec
        return oov_vector(word, self._dim)

    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self._dim
        vectors = [self._word_vector(tok) for tok in tokens]
        mean = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            return [0.0] * self._dim
        return (mean / norm).tolist()


class OpenAICompatibleEmbedder:
    """OpenAI 兼容 embeddings API（如 Ollama ``/v1/embeddings``）。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigError("未设置 EMBEDDING_API_KEY（或复用的 LLM_API_KEY）。")
        if dim <= 0:
            raise ValueError(f"EMBEDDING_DIM 必须为正数，实际为 {dim}")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # 空串单独处理，避免部分服务拒识
        nonempty_idx = [i for i, t in enumerate(texts) if t.strip()]
        out: list[list[float] | None] = [None] * len(texts)
        for i, t in enumerate(texts):
            if not t.strip():
                out[i] = [0.0] * self._dim
        if not nonempty_idx:
            return [v if v is not None else [0.0] * self._dim for v in out]

        payload = [texts[i] for i in nonempty_idx]
        batch_size = 100
        by_index: dict[int, list[float]] = {}
        
        try:
            for batch_start in range(0, len(payload), batch_size):
                batch_payload = payload[batch_start:batch_start + batch_size]
                response = self._client.embeddings.create(model=self._model, input=batch_payload)
                for item in response.data:
                    by_index[batch_start + item.index] = item.embedding
        except APIError as exc:
            raise LLMError(f"Embedding 请求失败：{exc}") from exc
        except Exception as exc:
            raise LLMError(f"Embedding 请求失败：{exc}") from exc

        for j, src_i in enumerate(nonempty_idx):
            raw = by_index.get(j)
            if raw is None:
                raise LLMError("Embedding 响应缺少向量")
            out[src_i] = _l2_normalize(list(raw), self._dim)
        return [v if v is not None else [0.0] * self._dim for v in out]


def _l2_normalize(vec: list[float], expected_dim: int) -> list[float]:
    if len(vec) != expected_dim:
        raise LLMError(
            f"Embedding 维度不匹配：期望 {expected_dim}，实际 {len(vec)}。"
            "请检查 EMBEDDING_DIM / EMBEDDING_MODEL。"
        )
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return [0.0] * expected_dim
    return (arr / norm).tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
