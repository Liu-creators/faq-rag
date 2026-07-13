"""统一的应用配置，从环境变量加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# DeepSeek OpenAI 兼容接口默认值（https://api-docs.deepseek.com/）。
# 若使用其他网关（如 Ollama），请覆盖 LLM_BASE_URL。
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"

DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_COLLECTION = "faq_questions"

DEFAULT_MYSQL_HOST = "127.0.0.1"
DEFAULT_MYSQL_PORT = 3306
DEFAULT_MYSQL_USER = "faq"
DEFAULT_MYSQL_PASSWORD = "faq"
DEFAULT_MYSQL_DATABASE = "faq_rag"


@dataclass(frozen=True)
class Settings:
    """运行时配置。LLM 字段供改写与后续文档 RAG 共用。"""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    word2vec_model_path: str
    milvus_uri: str
    milvus_token: str
    milvus_collection: str
    milvus_enabled: bool
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    database_url: str

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def word2vec_configured(self) -> bool:
        return bool(self.word2vec_model_path.strip())

    @property
    def milvus_configured(self) -> bool:
        return self.milvus_enabled and bool(self.milvus_uri.strip())

    @property
    def openai_embedding_configured(self) -> bool:
        return bool(self.embedding_model.strip()) and self.embedding_dim > 0


def _build_database_url(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> str:
    # mysql+pymysql://user:pass@host:port/db?charset=utf8mb4
    from urllib.parse import quote_plus

    return (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}?charset=utf8mb4"
    )


@lru_cache
def get_settings() -> Settings:
    milvus_flag = os.getenv("MILVUS_ENABLED", "true").strip().lower()
    milvus_enabled = milvus_flag in {"1", "true", "yes", "on"}

    embedding_dim_raw = os.getenv("EMBEDDING_DIM", "").strip()
    embedding_dim = int(embedding_dim_raw) if embedding_dim_raw else 0

    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if not embedding_api_key:
        embedding_api_key = os.getenv("LLM_API_KEY", "").strip()

    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/")
    if not embedding_base_url:
        embedding_base_url = (
            os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip().rstrip("/")
            or DEFAULT_LLM_BASE_URL
        )

    mysql_host = os.getenv("MYSQL_HOST", DEFAULT_MYSQL_HOST).strip() or DEFAULT_MYSQL_HOST
    mysql_port_raw = os.getenv("MYSQL_PORT", str(DEFAULT_MYSQL_PORT)).strip()
    mysql_port = int(mysql_port_raw) if mysql_port_raw else DEFAULT_MYSQL_PORT
    mysql_user = os.getenv("MYSQL_USER", DEFAULT_MYSQL_USER).strip() or DEFAULT_MYSQL_USER
    mysql_password = os.getenv("MYSQL_PASSWORD", DEFAULT_MYSQL_PASSWORD)
    mysql_database = (
        os.getenv("MYSQL_DATABASE", DEFAULT_MYSQL_DATABASE).strip() or DEFAULT_MYSQL_DATABASE
    )

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        database_url = _build_database_url(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )

    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip().rstrip("/")
        or DEFAULT_LLM_BASE_URL,
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL,
        word2vec_model_path=os.getenv("WORD2VEC_MODEL_PATH", "").strip(),
        milvus_uri=os.getenv("MILVUS_URI", DEFAULT_MILVUS_URI).strip() or DEFAULT_MILVUS_URI,
        milvus_token=os.getenv("MILVUS_TOKEN", "").strip(),
        milvus_collection=os.getenv("MILVUS_COLLECTION", DEFAULT_MILVUS_COLLECTION).strip()
        or DEFAULT_MILVUS_COLLECTION,
        milvus_enabled=milvus_enabled,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
        embedding_dim=embedding_dim,
        mysql_host=mysql_host,
        mysql_port=mysql_port,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_database=mysql_database,
        database_url=database_url,
    )
