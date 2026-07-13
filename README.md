# FAQ RAG

基于 FastAPI 的 FAQ + 文档 RAG 问答服务：对高频、口径确定的问题给出稳定答案，对长尾问题用文档检索增强生成兜底。

## 项目介绍

本项目面向「客服 / 知识问答」类场景，把知识拆成两层：

| 层级 | 定位 | 特点 |
|------|------|------|
| **FAQ** | 用户提问后的**第一步**处理 | 标准问 + 相似问 + 审定答案；命中后答案稳定、可审计 |
| **文档 RAG** | **粗粒度 / 非高频**知识兜底 | 覆盖手册、说明类长尾问题；允许生成，但不替代 FAQ 的确定性 |

FAQ 不是唯一引擎，而是旁路式的确定性答案层：优先用 FAQ 锁口径，FAQ 不够再用文档 RAG 补覆盖。

当前已具备 FAQ CRUD（**MySQL 持久化**）、一阶段检索（默认 **Milvus 混合检索**：稠密向量 + BM25 / RRF）、置信度分流，以及中置信区间的 FAQ 锚定 LLM 改写；文档 RAG 仍为占位。

## 整体流程

```text
用户提问
    │
    ▼
┌───────────────────┐
│  1. FAQ 检索      │  ← Milvus：dense + BM25 混合召回
└─────────┬─────────┘
          │
          ├─ 置信度极高 ──► 原样返回标准答案（不改写）
          │
          ├─ 置信度高   ──► 基于 FAQ 答案改写（事实锁定）
          │
          └─ 未命中/偏低
                    │
                    ▼
          ┌───────────────────┐
          │  2. 文档 RAG      │  ← 粗粒度 / 长尾兜底
          │  检索文档片段     │
          │  + LLM 生成回答   │
          └───────────────────┘
```

要点：

1. **FAQ 优先**：确定性问题走 FAQ，保证答案稳定。
2. **按置信度分流**：极高 → 不改写；高 → 可改写但锚定 FAQ 事实；低/未命中 → 文档 RAG。
3. **文档 RAG 兜底**：非高频、说明性内容，不与 FAQ 抢主路径。
4. **检索层**：默认 Milvus 混合检索（向量语义 + BM25 关键词）；可回退到内存 Word2Vec 或 ExactContainment。

## 环境

使用 conda 虚拟环境 `faq-rag`（Python 3.12+）。

```bash
conda activate faq-rag
pip install -e .
cp .env.example .env   # 填写 LLM_API_KEY；并配置稠密向量来源
```

### 启动依赖（MySQL + Milvus）

项目根目录 `docker-compose.yml` 同时提供 **MySQL**（FAQ 持久化）与 **Milvus**（混合检索；内置 BM25 需完整 Milvus ≥ 2.5）：

```bash
docker compose up -d
docker compose ps
# MySQL：localhost:3306 / 库 faq_rag（用户 faq / 密码 faq）
# Milvus：http://localhost:19530
```

### 配置说明

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `MYSQL_HOST` | 否 | `127.0.0.1` | MySQL 主机 |
| `MYSQL_PORT` | 否 | `3306` | MySQL 端口 |
| `MYSQL_USER` | 否 | `faq` | MySQL 用户 |
| `MYSQL_PASSWORD` | 否 | `faq` | MySQL 密码 |
| `MYSQL_DATABASE` | 否 | `faq_rag` | 数据库名 |
| `DATABASE_URL` | 否 | （由上面拼出） | 完整 SQLAlchemy URL，设置后优先 |
| `LLM_API_KEY` | 是（走改写时） | — | DeepSeek API Key |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容 Base URL |
| `LLM_MODEL` | 否 | `deepseek-v4-flash` | 模型名 |
| `MILVUS_ENABLED` | 否 | `true` | 是否启用 Milvus 混合检索 |
| `MILVUS_URI` | 否 | `http://localhost:19530` | Milvus 地址 |
| `MILVUS_TOKEN` | 否 | （空） | 认证 token（如有） |
| `MILVUS_COLLECTION` | 否 | `faq_questions` | 集合名 |
| `WORD2VEC_MODEL_PATH` | 二选一* | — | 稠密向量：预训 `.kv` |
| `EMBEDDING_MODEL` + `EMBEDDING_DIM` | 二选一* | — | 稠密向量：OpenAI 兼容 embeddings |

\* 启用 Milvus 时必须配置其一：`WORD2VEC_MODEL_PATH`，或 `EMBEDDING_MODEL` + `EMBEDDING_DIM`（可选 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`，默认回落到 LLM 相关变量）。

关闭 Milvus（`MILVUS_ENABLED=false`）时：有 Word2Vec 则用内存线性扫描；否则 ExactContainment。

### Word2Vec 稠密向量（本地起步）

```bash
python scripts/train_word2vec.py \
  --input data/raw/ollama_docs \
  --output data/models/word2vec.kv

# .env
WORD2VEC_MODEL_PATH=data/models/word2vec.kv
```

混合检索流程：Milvus 用 **dense ANN + 中文 BM25** 做 RRF 召回；对外置信度仍用**稠密余弦**（兼容现有 0.90 / 0.70 阈值）。

## 启动

```bash
conda activate faq-rag
which python   # 应指向 .../miniconda3/envs/faq-rag/bin/python

# 先确保 MySQL + Milvus 已起
docker compose up -d

python main.py
# 或
uvicorn faq_rag.main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问：

- API 根路径：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- 提问入口：http://127.0.0.1:8000/ask
- 交互文档：http://127.0.0.1:8000/docs

## FAQ CRUD（MySQL 持久化）

FAQ 业务数据存在 MySQL 表 `faqs`（标准问 / 答案 / 相似问法 JSON）。应用启动时自动 `create_all`；Milvus 仍只索引问法向量。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/faqs` | 新建 FAQ |
| GET | `/faqs` | 列表（支持 `q` / `category` / `enabled`） |
| GET | `/faqs/{id}` | 详情 |
| PATCH | `/faqs/{id}` | 部分更新 |
| DELETE | `/faqs/{id}` | 删除 |

示例：

```bash
curl -s http://127.0.0.1:8000/faqs -H 'Content-Type: application/json' -d '{
  "question": "如何安装 Ollama？",
  "answer": "参见官方安装文档，按系统选择安装方式。",
  "similar_questions": ["Ollama 怎么装", "安装 ollama"],
  "category": "install"
}'
```

## Ask（提问入口）

`POST /ask`：先 FAQ 检索，再按置信度分流。中置信走 FAQ 锚定 LLM 改写；文档 RAG 目前返回占位文案。

当前阈值：≥ 0.90 → 原样返回；≥ 0.70 → FAQ 锚定 LLM 改写；否则 → 文档 RAG 占位。

## 项目结构

```text
main.py                        # 根入口
docker-compose.yml             # MySQL + Milvus standalone
scripts/                       # 文档采集 / Word2Vec 训练
data/raw/ollama_docs/          # 原始文档语料
src/faq_rag/
  config.py                    # 环境变量配置
  db/                          # SQLAlchemy / MySQL
  services/
    faq/                       # FAQ 存储与检索
    similarity/                # 相似度索引
      milvus_index.py          # Milvus 混合检索（dense + BM25）
      embedder.py              # Word2Vec / OpenAI 兼容稠密编码
      word2vec.py              # 内存 Word2Vec 回退
      index.py                 # Protocol + ExactContainment
    ask/                       # 问答流水线
  api/routes/                  # HTTP 路由
```

## 当前进度

- [x] FAQ CRUD（MySQL 持久化）
- [x] FAQ 旁路门控骨架（极高置信原样 / 高置信改写 / 否则文档 RAG 占位）
- [x] FAQ 改写（LLM，事实锁定）
- [x] Milvus 混合检索（稠密向量 + 中文 BM25 / RRF）
- [x] 稠密编码器（Word2Vec 或 OpenAI 兼容 embeddings）
- [ ] FAQ CRUD 时增量 upsert（当前仍为查询前全量同步）
- [ ] 文档 RAG 粗粒度兜底
