"""Milvus 混合检索：稠密向量 + 内置 BM25，RRF 融合后再用稠密余弦打分。"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from faq_rag.services.similarity.embedder import DenseEmbedder, cosine_similarity
from faq_rag.services.similarity.index import IndexedDocument, SimilarityHit

logger = logging.getLogger(__name__)

_PK_MAX_LEN = 64
_TEXT_MAX_LEN = 2000
_DOC_ID_MAX_LEN = 64


def _entry_pk(doc_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}\0{text}".encode("utf-8")).hexdigest()
    return digest[:_PK_MAX_LEN]


def _escape_filter_str(value: str) -> str:
    """转义 Milvus 布尔表达式中的字符串字面量。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _dense_dim_from_describe(info: object) -> int | None:
    fields = None
    if isinstance(info, dict):
        fields = info.get("fields")
    else:
        fields = getattr(info, "fields", None)
    if not fields:
        return None
    for field in fields:
        if isinstance(field, dict):
            name = field.get("name")
            params = field.get("params") or {}
            dim = params.get("dim", field.get("dim"))
        else:
            name = getattr(field, "name", None)
            params = getattr(field, "params", None) or {}
            dim = params.get("dim") if isinstance(params, dict) else None
            if dim is None:
                dim = getattr(field, "dim", None)
        if name == "dense_vector" and dim is not None:
            return int(dim)
    return None


class MilvusHybridIndex:
    """Milvus 上的 FAQ 混合索引（dense ANN + BM25，RRF 召回）。

    置信度分数使用稠密向量余弦相似度（与既有 0.90 / 0.70 阈值兼容），
    而非原始 RRF 分。BM25 仅参与召回融合。

    需要 Milvus ≥ 2.5（内置 BM25）；Milvus Lite 不支持该功能。
    """

    def __init__(
        self,
        *,
        uri: str,
        token: str = "",
        collection: str,
        embedder: DenseEmbedder,
        search_limit_multiplier: int = 3,
    ) -> None:
        if not uri.strip():
            raise ValueError("MILVUS_URI 不能为空")
        if not collection.strip():
            raise ValueError("MILVUS_COLLECTION 不能为空")
        if search_limit_multiplier < 1:
            raise ValueError("search_limit_multiplier 必须 >= 1")

        self._uri = uri.strip()
        self._token = token.strip()
        self._collection = collection.strip()
        self._embedder = embedder
        self._search_limit_multiplier = search_limit_multiplier
        self._client: MilvusClient | None = None

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            kwargs: dict = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            self._client = MilvusClient(**kwargs)
        return self._client

    def _ensure_collection(self) -> None:
        client = self._get_client()
        dim = self._embedder.dim
        if client.has_collection(self._collection):
            # 维度变更时重建；BM25 / analyzer 也无法在线改 schema
            info = client.describe_collection(self._collection)
            existing_dim = _dense_dim_from_describe(info)
            if existing_dim == dim:
                client.load_collection(self._collection)
                return
            logger.warning(
                "集合 %s 的 dense 维度为 %s，与当前 embedder(%s) 不一致，将重建",
                self._collection,
                existing_dim,
                dim,
            )
            client.drop_collection(self._collection)

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_PK_MAX_LEN,
        )
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            max_length=_DOC_ID_MAX_LEN,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=_TEXT_MAX_LEN,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        client.load_collection(self._collection)

    def _prepare_rows(self, documents: Sequence[IndexedDocument]) -> list[dict]:
        texts: list[str] = []
        meta: list[tuple[str, str, str]] = []  # pk, doc_id, text
        for doc in documents:
            text = doc.text.strip()
            if not text:
                continue
            if len(text) > _TEXT_MAX_LEN:
                text = text[:_TEXT_MAX_LEN]
            if len(doc.doc_id) > _DOC_ID_MAX_LEN:
                logger.warning("doc_id 过长已截断：%s", doc.doc_id)
            doc_id = doc.doc_id[:_DOC_ID_MAX_LEN]
            pk = _entry_pk(doc_id, text)
            meta.append((pk, doc_id, text))
            texts.append(text)

        if not meta:
            return []

        vectors = self._embedder.embed(texts)
        return [
            {
                "pk": pk,
                "doc_id": doc_id,
                "text": text,
                "dense_vector": dense,
            }
            for (pk, doc_id, text), dense in zip(meta, vectors, strict=True)
        ]

    def build(self, documents: Sequence[IndexedDocument]) -> None:
        """全量替换集合内容（启动 / 修复用）。"""
        client = self._get_client()
        self._ensure_collection()
        # 清空旧数据：过滤删掉所有主键非空行
        try:
            client.delete(collection_name=self._collection, filter='pk != ""')
        except Exception as exc:  # noqa: BLE001 — 空集合等可忽略
            logger.debug("清空集合时忽略：%s", exc)

        rows = self._prepare_rows(documents)
        if not rows:
            return

        client.upsert(collection_name=self._collection, data=rows)
        client.flush(self._collection)
        client.load_collection(self._collection)

    def upsert(self, documents: Sequence[IndexedDocument]) -> None:
        """增量写入；同 pk 覆盖，不清理同 doc_id 下已删除的旧文本。"""
        rows = self._prepare_rows(documents)
        if not rows:
            return

        client = self._get_client()
        self._ensure_collection()
        client.upsert(collection_name=self._collection, data=rows)
        client.flush(self._collection)
        client.load_collection(self._collection)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除该 FAQ 下全部问题行。"""
        client = self._get_client()
        if not client.has_collection(self._collection):
            return
        safe_id = _escape_filter_str(doc_id[:_DOC_ID_MAX_LEN])
        try:
            client.delete(
                collection_name=self._collection,
                filter=f'doc_id == "{safe_id}"',
            )
            client.flush(self._collection)
        except Exception as exc:  # noqa: BLE001 — 空集合 / 无匹配可忽略
            logger.debug("按 doc_id 删除时忽略：%s", exc)

    def search(self, query: str, *, top_k: int = 1) -> list[SimilarityHit]:
        if top_k <= 0 or not query.strip():
            return []

        query_text = query.strip()
        if len(query_text) > _TEXT_MAX_LEN:
            query_text = query_text[:_TEXT_MAX_LEN]

        query_vec = self._embedder.embed([query_text])[0]
        if all(v == 0.0 for v in query_vec):
            return []

        candidate_limit = max(top_k * self._search_limit_multiplier, top_k)

        dense_req = AnnSearchRequest(
            data=[query_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=candidate_limit,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=candidate_limit,
        )

        client = self._get_client()
        self._ensure_collection()
        raw = client.hybrid_search(
            collection_name=self._collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=candidate_limit,
            output_fields=["doc_id", "text", "dense_vector"],
        )

        hits_batch = raw[0] if raw else []
        best_by_doc: dict[str, SimilarityHit] = {}
        for hit in hits_batch:
            entity = hit.get("entity") or {}
            doc_id = str(entity.get("doc_id") or hit.get("doc_id") or "")
            text = str(entity.get("text") or "")
            dense = entity.get("dense_vector")
            if not doc_id or not text:
                continue
            if dense is None:
                score = 0.0
            else:
                score = cosine_similarity(query_vec, dense)
            if score <= 0:
                continue
            prev = best_by_doc.get(doc_id)
            if prev is None or score > prev.score:
                best_by_doc[doc_id] = SimilarityHit(
                    doc_id=doc_id,
                    text=text,
                    score=score,
                )

        ranked = sorted(best_by_doc.values(), key=lambda h: h.score, reverse=True)
        return ranked[:top_k]

    def has_data(self) -> bool:
        client = self._get_client()
        if not client.has_collection(self._collection):
            return False
        try:
            res = client.query(
                collection_name=self._collection,
                filter='pk != ""',
                limit=1,
                output_fields=["pk"]
            )
            return len(res) > 0
        except Exception as exc:
            logger.debug("检查 Milvus 是否存在数据时出错：%s", exc)
            return False
