# FAQ RAG

基于 FastAPI 的 FAQ + 文档 RAG 问答服务：对高频、口径确定的问题给出稳定答案，对长尾问题用文档检索增强生成兜底。

## 项目介绍

本项目面向「客服 / 知识问答」类场景，把知识拆成两层：

| 层级 | 定位 | 特点 |
|------|------|------|
| **FAQ** | 用户提问后的**第一步**处理 | 标准问 + 相似问 + 审定答案；命中后答案稳定、可审计 |
| **文档 RAG** | **粗粒度 / 非高频**知识兜底 | 覆盖手册、说明类长尾问题；允许生成，但不替代 FAQ 的确定性 |

FAQ 不是唯一引擎，而是旁路式的确定性答案层：优先用 FAQ 锁口径，FAQ 不够再用文档 RAG 补覆盖。

当前已具备 FAQ CRUD（**MySQL 持久化**）、一阶段检索（默认 **Milvus 混合检索**：稠密向量 + BM25 / RRF）、置信度分流，中置信区间的 FAQ 锚定 LLM 改写，以及文档 RAG 粗粒度兜底（切块入库 + 混合检索 + 带引用生成）。

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

## FAQ 管理流程

业务数据与检索索引分离：**MySQL** 存标准问 / 答案 / 相似问 / 元数据；**Milvus** 只索引问法文本（默认；关闭后回退 Word2Vec / ExactContainment）。CRUD 写库后按需增量同步 Milvus。

### 数据落点

| 存储 | 存什么 | 不存什么 |
|------|--------|----------|
| **MySQL `faqs`** | `id` / `question` / `answer` / `similar_questions` / `category` / `enabled` / 时间戳 | 向量 |
| **Milvus `faq_questions`** | 每条问法一行：`pk`、`doc_id`（= FAQ id）、`text`、`dense_vector`、`sparse_vector`（BM25 自动生成） | 答案、类目等业务字段 |
| **MySQL `documents` / `doc_chunks`** | 文档元数据与切块正文 | 向量 |
| **Milvus `doc_chunks`** | 每条切块一行：`pk`、`doc_id`（= chunk id）、`text`、dense / sparse | 文档标题等业务字段 |

一条 FAQ 会展开为 **1（标准问）+ N（相似问）** 行；主键 `pk = sha256(doc_id + text)`。

### CRUD → MySQL + Milvus

```text
                    ┌─────────────────────────────────────┐
                    │           FAQ CRUD API              │
                    │  POST / GET / PATCH / DELETE /faqs  │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
         新建 / 更新              列表 / 详情              删除
              │                       │                       │
              ▼                       ▼                       ▼
      ┌───────────────┐        ┌───────────────┐      ┌───────────────┐
      │ MySQL faqs 表 │        │ 仅读 MySQL    │      │ MySQL 删除行  │
      │ 写入/更新     │        │ （不碰 Milvus）│      └───────┬───────┘
      └───────┬───────┘        └───────────────┘              │
              │                                               ▼
              ▼                                    ┌─────────────────────┐
      影响问法？                                   │ Milvus               │
   （question /                                    │ delete filter:       │
    similar_questions /                            │   doc_id == faq_id   │
    enabled）                                      │ flush                │
              │                                    └─────────────────────┘
     ┌────────┴────────┐
     │ 否              │ 是
     ▼                 ▼
  结束          ┌──────────────────────────────────────────┐
                │ Milvus 增量同步（upsert_faq）             │
                │                                          │
                │ 1. delete_by_doc_id(faq_id)              │
                │    → 清掉该 FAQ 下全部旧问法行            │
                │                                          │
                │ 2. 若 enabled=false → 结束（只删不写）    │
                │                                          │
                │ 3. 否则展开问法 → 稠密编码 embed()       │
                │    → upsert 行（text + dense_vector）    │
                │    → Milvus 对 text 自动算 BM25 sparse   │
                │    → flush + load_collection             │
                └──────────────────────────────────────────┘
```

### 全量引导 / 修复（`sync_index` → Milvus `build`）

```text
首次检索 或 增量失败（_ready=false）
    │
    ▼
从 MySQL 拉取 enabled=true 的 FAQ
展开全部标准问 + 相似问
    │
    ▼
Milvus build：
  1. ensure_collection（无集合则建：dense COSINE + sparse BM25 / 中文 analyzer；
     dense 维度与当前 embedder 不一致则 drop 重建）
  2. delete 清空集合内旧行
  3. embed 全部问法 → upsert
  4. flush + load
    │
    ▼
之后 CRUD 走上面的增量维护
```

要点：

1. **双写职责**：MySQL 是真相源；Milvus 只服务问法召回（dense + BM25），不含答案。
2. **先删后写**：增量同步总是先按 `doc_id` 删干净，再写入当前问法，避免相似问改少后留下幽灵行。
3. **增量触发**：新建必同步；更新仅当 `question` / `similar_questions` / `enabled` 变化时同步；删除必从 Milvus 移除。
4. **禁用即下线**：`enabled=false` 时只删 Milvus 行；检索侧也会跳过未启用 FAQ。
5. **懒引导**：索引未就绪时 CRUD 跳过 Milvus；下次检索全量 `build`（已含当前库数据）；增量异常同样标记未就绪后全量修复。

## 环境

使用 conda 虚拟环境 `faq-rag`（Python 3.12+）。

```bash
conda activate faq-rag
pip install -e .
cp .env.example .env   # 填写 LLM_API_KEY；默认已配 Ollama qwen3-embedding:0.6b
```

### 启动依赖（MySQL + Milvus + Attu）

项目根目录 `docker-compose.yml` 同时提供 **MySQL**（FAQ 持久化）、**Milvus**（混合检索；内置 BM25 需完整 Milvus ≥ 2.5）与 **Attu**（向量库 Web 管理界面）：

```bash
docker compose up -d
docker compose ps
# MySQL：localhost:3306 / 库 faq_rag（用户 faq / 密码 faq）
# Milvus：http://localhost:19530
# Attu：http://localhost:3000
```

### 查看向量库（Milvus）

本项目**没有自带向量库前端**；FAQ / 文档的业务数据在 MySQL，向量索引在 Milvus。浏览集合与向量数据用官方管理工具 **[Attu](https://milvus.io/docs/attu.md)**（已随 `docker-compose.yml` 启动）。

1. 打开 http://localhost:3000
2. 连接地址填 `standalone:19530`（Attu 容器已通过环境变量 `MILVUS_URL` 预填；若手动连接，勿填 `localhost`，容器内访问不到宿主机的 loopback）
3. 常见集合：
   - `faq_questions`：FAQ 问法（标准问 + 相似问）向量行
   - `doc_chunks`：文档切块向量行

也可用 Python 快速核对（需已 `pip install -e .` 且 Milvus 已起）：

```bash
python - <<'PY'
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
print("collections:", c.list_collections())
for name in c.list_collections():
    print(name, "->", c.get_collection_stats(name))
PY
```

说明：

- http://127.0.0.1:8000/docs 是本服务的 **FastAPI Swagger**，不是向量库 UI。
- http://localhost:9001 是 **MinIO 控制台**（Milvus 底层对象存储），一般不必用来看 FAQ / 文档向量。
- 本仓库 Milvus 为 **2.5.x**，Attu 镜像需用 **v2.5**（`zilliz/attu:v2.5`）；Attu v3 面向 Milvus 3.x，不兼容当前版本。

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
| `MILVUS_COLLECTION` | 否 | `faq_questions` | FAQ 问法集合名 |
| `MILVUS_DOC_COLLECTION` | 否 | `doc_chunks` | 文档切块集合名 |
| `DOC_RAW_DIR` | 否 | `data/raw/ollama_docs` | 文档入库默认目录 |
| `DOC_RAG_TOP_K` | 否 | `5` | 文档 RAG 召回条数 |
| `DOC_CHUNK_SIZE` | 否 | `500` | Markdown 切块目标长度（字符） |
| `DOC_CHUNK_OVERLAP` | 否 | `80` | 切块重叠长度（字符） |
| `WORD2VEC_MODEL_PATH` | 二选一* | — | 稠密向量：预训 `.kv`（离线回退） |
| `EMBEDDING_MODEL` + `EMBEDDING_DIM` | 二选一* | — | 稠密向量：OpenAI 兼容 embeddings（推荐） |
| `EMBEDDING_BASE_URL` | 否 | 回落 `LLM_BASE_URL` | Embedding API Base URL（Ollama 示例见下） |
| `EMBEDDING_API_KEY` | 否 | 回落 `LLM_API_KEY` | Embedding API Key（Ollama 可填 `ollama`） |

\* 启用 Milvus 时必须配置其一：`EMBEDDING_MODEL` + `EMBEDDING_DIM`，或 `WORD2VEC_MODEL_PATH`。`.env.example` 默认走 Ollama embedding。

关闭 Milvus（`MILVUS_ENABLED=false`）时：有 Word2Vec 则用内存线性扫描；否则 ExactContainment。

### 文本向量化（稠密编码器）

Word2Vec 与 OpenAI 兼容 embeddings 做的是同一件事：**把文本编码成稠密向量**，供相似度检索（Milvus dense 通道，或关闭 Milvus 时的内存扫描）。二者实现同一接口，配置上二选一即可。

| | OpenAI 兼容 embeddings（推荐） | Word2Vec（本地 `.kv`） |
|--|-------------------------------|------------------------|
| 配置 | `EMBEDDING_MODEL` + `EMBEDDING_DIM`（可选 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`） | `WORD2VEC_MODEL_PATH` |
| 做法 | 整句/整段直接编码 | jieba 分词 → 词向量均值 |
| 部署 | 调 embedding API（如 Ollama / 云厂商） | 本地文件，无额外服务 |
| 适用 | 中英 FAQ / 文档语义检索 | 完全离线、快速冒烟 |

**选择优先级**（`build_dense_embedder`）：

1. 若配置了 `EMBEDDING_MODEL` + `EMBEDDING_DIM` → 用 OpenAI 兼容 embeddings
2. 否则若配置了 `WORD2VEC_MODEL_PATH` → 用 Word2Vec
3. 启用 Milvus 时两者都缺 → 启动/检索时报错

**检索后端优先级**（与编码器独立）：

1. Milvus 开启且 URI 可用 → 混合检索（dense + BM25 / RRF）；dense 通道用上面选中的编码器
2. 否则有 Word2Vec → 内存 `Word2VecIndex` 线性扫描
3. 否则 → `ExactContainmentIndex`（字符串包含打分，不做向量化）

#### OpenAI 兼容 embeddings（推荐：Ollama）

本机已安装 [Ollama](https://ollama.com) 时，推荐使用 **`qwen3-embedding:0.6b`**（约 639MB，默认维度 1024）：中英语义较好，适合本项目的中文 FAQ + 英文文档场景。勿与 `LLM_BASE_URL`（如 DeepSeek）混用同一地址——embedding 与聊天 LLM 应分开配置。

```bash
ollama pull qwen3-embedding:0.6b

# 确认维度（应输出 1024）
curl -s http://localhost:11434/api/embed \
  -d '{"model":"qwen3-embedding:0.6b","input":"如何安装 Ollama？"}' \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))"

# .env（设置后优先于 Word2Vec；与 .env.example 一致）
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIM=1024
```

换模型或改 `EMBEDDING_DIM` 后，Milvus 若已有旧维度集合会 **drop 重建**，需重新同步 FAQ / 文档索引；置信度阈值 `0.90 / 0.70` 也可能需按新分数分布微调。

未单独设置时，`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` 会回落到 `LLM_BASE_URL` / `LLM_API_KEY`。

#### Word2Vec（离线回退）

```bash
python scripts/train_word2vec.py \
  --input data/raw/ollama_docs \
  --output data/models/word2vec.kv

# .env（不要同时依赖 API embedding 时再用）
WORD2VEC_MODEL_PATH=data/models/word2vec.kv
```

混合检索流程：Milvus 用 **dense ANN + 中文 BM25** 做 RRF 召回；对外置信度仍用**稠密余弦**（兼容现有 0.90 / 0.70 阈值）。

## 启动

```bash
conda activate faq-rag
which python   # 应指向 .../miniconda3/envs/faq-rag/bin/python

# 安装 / 同步依赖（首次，或 pyproject.toml 变更后必跑）
pip install -e .

# 先确保 MySQL + Milvus（及 Attu）已起
docker compose up -d

python main.py
# 或
uvicorn faq_rag.main:app --reload --host 0.0.0.0 --port 8000
```

若出现 `ModuleNotFoundError`（如 `sqlalchemy` / `pymysql`），多半是依赖未装全或未同步，在已激活的 `faq-rag` 环境中重新执行 `pip install -e .` 即可。

启动后访问：

- API 根路径：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- 提问入口：http://127.0.0.1:8000/ask
- 交互文档：http://127.0.0.1:8000/docs

## 数据初始化

服务起来后，需要先把语料入库才能问答。FAQ 用 [CRUD API](#faq-crudmysql-持久化) 写入；文档 RAG 走 **采集 → 清洗 → 入库**，细节见仓库内 [`scripts/README.md`](scripts/README.md)：

- [文档 RAG 数据流水线](scripts/README.md#文档-rag-数据流水线)
- [采集 Ollama 官方文档](scripts/README.md#采集-ollama-官方文档)
- [清洗 Ollama 文档](scripts/README.md#清洗-ollama-文档文档-rag)
- [文档入库（MySQL + Milvus）](scripts/README.md#文档入库mysql--milvus)
- [训练 Word2Vec（可选离线回退）](scripts/README.md#训练-word2vecfaq-相似度)

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

`POST /ask`：先 FAQ 检索，再按置信度分流。中置信走 FAQ 锚定 LLM 改写；低置信 / 未命中走文档 RAG（检索切块 + LLM 生成，响应中带 `sources`）。

当前阈值：≥ 0.90 → 原样返回；≥ 0.70 → FAQ 锚定 LLM 改写；否则 → 文档 RAG。

## 文档 RAG

业务数据与检索索引分离：**MySQL** 存文档元数据与切块；**Milvus** `doc_chunks` 集合索引切块文本（dense + BM25）。入库时按 Markdown 标题切块，块内再按字符窗口切分。

文档 RAG **并非天生必须 MySQL**（也可只把正文放进向量库 payload）。本项目与 FAQ 对齐，采用 **MySQL 真相源 + Milvus 召回**：便于列表/禁用、换模型后从库重建索引，以及生成时回表取全文与引用。代价是双写、查询多一跳、存储有冗余。流水线与取舍说明见 [`scripts/README.md`](scripts/README.md)。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/documents/ingest` | 扫描本地 Markdown 目录入库并重建文档索引 |
| GET | `/documents` | 文档列表（支持 `category` / `enabled`） |
| POST | `/documents/reindex` | 仅从 MySQL 重建 Milvus 文档索引 |

示例：

```bash
# 采集 → 清洗 → 入库（推荐）
python scripts/collect_ollama_docs.py
python scripts/clean_ollama_docs.py
python scripts/ingest_docs.py --directory data/cleaned/ollama_docs --category ollama

# 或服务已启动时用 HTTP
curl -s http://127.0.0.1:8000/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"directory": "data/cleaned/ollama_docs", "category": "ollama", "rebuild_index": true}'
```

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
    doc/                       # 文档入库 / 切块 / 检索
    similarity/                # 相似度索引
      milvus_index.py          # Milvus 混合检索（dense + BM25）
      embedder.py              # Word2Vec / OpenAI 兼容稠密编码
      word2vec.py              # 内存 Word2Vec 回退
      index.py                 # Protocol + ExactContainment
    ask/                       # 问答流水线（含文档 RAG 生成）
  api/routes/                  # HTTP 路由
```

## 当前进度

- [x] FAQ CRUD（MySQL 持久化）
- [x] FAQ 旁路门控骨架（极高置信原样 / 高置信改写 / 否则文档 RAG）
- [x] FAQ 改写（LLM，事实锁定）
- [x] Milvus 混合检索（稠密向量 + 中文 BM25 / RRF）
- [x] 稠密编码器（Word2Vec 或 OpenAI 兼容 embeddings）
- [x] FAQ CRUD 时增量 upsert（首次检索全量引导，之后 CRUD 增量维护）
- [x] 文档 RAG 粗粒度兜底（切块入库 + 混合检索 + 带引用生成）
