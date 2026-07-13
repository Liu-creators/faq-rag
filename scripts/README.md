# Scripts

## 文档 RAG 数据流水线

从官方文档站到可检索知识库的主路径：**采集 → 清洗 → 入库**。

```text
docs.ollama.com
      │
      ▼
collect_ollama_docs.py
      │
      ▼
data/raw/ollama_docs/          （原始 Markdown + manifest.jsonl）
      │
      ▼
clean_ollama_docs.py
      │
      ▼
data/cleaned/ollama_docs/      （纯 Markdown，适合 RAG）
      │
      ▼
ingest_docs.py
（或 POST /documents/ingest）
      │
      ▼
切块 chunk_markdown
      │
      ├──────────────────────┐
      ▼                      ▼
MySQL                      Milvus doc_chunks
documents / doc_chunks     dense + BM25
（真相源：正文/元数据）      （召回索引）
      │                      ▲
      └──── sync_index ──────┘
            （可从 MySQL 重建）
```

查询时：Milvus 召回切块 id → MySQL 取正文与文档元数据 → LLM 生成并带引用。

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 采集 | `collect_ollama_docs.py` | 官方文档站 | `data/raw/ollama_docs` |
| 清洗 | `clean_ollama_docs.py` | raw 目录 | `data/cleaned/ollama_docs` |
| 入库 | `ingest_docs.py` | cleaned 目录 | MySQL 正文 + Milvus 向量 |

### 为何文档 RAG 也双写 MySQL + Milvus

FAQ 侧 MySQL 存标准问 / 答案等业务字段，几乎是刚需；文档 RAG 理论上可以「只写 Milvus、payload 带正文」。本项目**刻意与 FAQ 共用同一套模式**：**MySQL 真相源 + Milvus 召回索引**，方便统一运维与重建。

| | MySQL `documents` / `doc_chunks` | Milvus `doc_chunks` |
|---|---|---|
| 存什么 | 标题、路径、类目、enabled、切块正文 | `text` + dense / BM25 sparse |
| 职责 | 真相源、列表/禁用、生成与引用时取全文 | 只负责「找谁相关」 |
| 可否丢失 | 不可（丢了要重新入库） | 可（`sync_index` / `/documents/reindex` 从 MySQL 重建） |

**优点**

- 索引可丢可重建：换 embedding 维度或集合损坏时，不必依赖清洗目录仍在本机。
- 管理能力完整：文档列表、`enabled` 下线、按类目过滤，与 FAQ CRUD 心智一致。
- 生成与引用稳定：召回只拿 id，再回表取完整切块与文档元数据。

**代价**

- 双写与短暂不一致：先写库再同步索引；增量失败时依赖下次全量重建。
- 查询多一跳：Milvus 命中后还要回 MySQL 水合正文。
- 对「纯文件 RAG Demo」略重：正文在库与索引中各存一份。

当前选择保留该模式；若日后只要轻量离线检索，再考虑改为仅 Milvus。

---

## 采集 Ollama 官方文档

从 [docs.ollama.com](https://docs.ollama.com) 抓取文档，保存为 Markdown，并生成 `manifest.jsonl` 索引，供后续 FAQ / RAG 入库使用。

### 依赖

脚本额外依赖（项目主依赖未包含）：

```bash
pip install requests beautifulsoup4 markdownify
```

### 用法

```bash
python scripts/collect_ollama_docs.py \
  --output-dir data/raw/ollama_docs \
  --max-pages 50 \
  --delay 1.5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `data/raw/ollama_docs` | 输出目录（Markdown + `manifest.jsonl`） |
| `--max-pages` | `100` | 最多保存的文档页数 |
| `--delay` | `1.5` | 请求间隔（秒），降低对目标站点压力 |
| `--timeout` | `20` | 单次 HTTP 超时（秒） |
| `--user-agent` | 内置标识 | 爬虫 User-Agent；正式使用前请改成自己的联系方式 |

### 输出

运行后目录结构大致如下：

```text
data/raw/ollama_docs/
  manifest.jsonl          # 每行一条文档元数据
  *.md                    # 文档正文（含 YAML front matter）
```

脚本会遵守 `robots.txt`，优先拉取页面的 `.md` 版本，不可用时再从 HTML 转 Markdown。

---

## 清洗 Ollama 文档（文档 RAG）

将 `data/raw/ollama_docs` 清洗为适合 **文档 RAG 入库** 的 Markdown（去掉 frontmatter / 横幅 / OpenAPI schema 噪声，修复转义与代码围栏）。

FAQ 页、整份 `openapi.yaml`、`api-reference` 重复页、站点首页默认跳过（FAQ 建议单独整理进 FAQ 表）。

### 用法

```bash
python scripts/clean_ollama_docs.py \
  --input-dir data/raw/ollama_docs \
  --output-dir data/cleaned/ollama_docs
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `data/raw/ollama_docs` | 采集原始目录 |
| `--output-dir` | `data/cleaned/ollama_docs` | 清洗输出目录 |
| `--min-chars` | `80` | 清洗后过短则跳过 |
| `--include-faq` | off | 保留 faq 页进 RAG |
| `--include-index` | off | 保留站点首页 |

### 输出

```text
data/cleaned/ollama_docs/
  manifest.jsonl    # 保留的文档元数据
  skipped.jsonl     # 跳过原因
  *.md              # 无 frontmatter 的纯 Markdown
```

---

## 文档入库（MySQL + Milvus）

将清洗后的 Markdown 切块写入 **MySQL**，并同步 **Milvus** `doc_chunks` 向量索引（与 `POST /documents/ingest` 同一套逻辑）。

### 前置

- `docker compose up -d`（MySQL + Milvus）
- `.env` 中 embedding / Milvus / MySQL 配置可用
- 建议先跑完清洗：`python scripts/clean_ollama_docs.py`

### 用法

```bash
python scripts/ingest_docs.py \
  --directory data/cleaned/ollama_docs \
  --category ollama
```

也可在服务已启动时用 HTTP：

```bash
curl -s http://127.0.0.1:8000/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"directory": "data/cleaned/ollama_docs", "category": "ollama"}'
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--directory` / `-d` | `data/cleaned/ollama_docs` | 清洗后的 Markdown 根目录 |
| `--category` / `-c` | （空） | 可选类目，写入本次全部文档 |
| `--no-rebuild-index` | off | 逐文件增量 upsert；默认入库后全量 `sync_index` |
| `--verbose` / `-v` | off | 打印每个文件的入库日志 |

---

## 训练 Word2Vec（FAQ 相似度）

在项目外（本脚本旁路）用 **jieba** 分词训练 gensim Word2Vec，只导出 `KeyedVectors`（`.kv`）供服务加载。分词与运行时共用 `faq_rag.services.similarity.tokenize.tokenize`。

### 依赖

需已安装项目依赖（含 `jieba` / `gensim` / `numpy`）：

```bash
pip install -e .
```

### 用法

```bash
python scripts/train_word2vec.py \
  --input data/raw/ollama_docs \
  --output data/models/word2vec.kv
```

也可传入多个文件或目录：

```bash
python scripts/train_word2vec.py \
  -i data/raw/corpus.txt data/raw/ollama_docs \
  -o data/models/word2vec.kv \
  --vector-size 100 \
  --window 5 \
  --min-count 1 \
  --epochs 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | （必填） | 语料文件或目录（可多个）；目录递归收集 `.txt` / `.md` |
| `--output` / `-o` | （必填） | 输出 `.kv` 路径 |
| `--vector-size` | `100` | 词向量维度 |
| `--window` | `5` | 上下文窗口 |
| `--min-count` | `1` | 最低词频 |
| `--epochs` | `10` | 训练轮数 |
| `--workers` | `4` | 并行线程数 |
| `--sg` | `0` | `0`=CBOW，`1`=skip-gram |

语料按行分句后 jieba 分词；空行跳过。训好后在 `.env` 设置：

```bash
WORD2VEC_MODEL_PATH=data/models/word2vec.kv
```
