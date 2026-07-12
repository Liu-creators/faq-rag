# FAQ RAG

基于 FastAPI 的 FAQ + 文档 RAG 问答服务：对高频、口径确定的问题给出稳定答案，对长尾问题用文档检索增强生成兜底。

## 项目介绍

本项目面向「客服 / 知识问答」类场景，把知识拆成两层：

| 层级 | 定位 | 特点 |
|------|------|------|
| **FAQ** | 用户提问后的**第一步**处理 | 标准问 + 相似问 + 审定答案；命中后答案稳定、可审计 |
| **文档 RAG** | **粗粒度 / 非高频**知识兜底 | 覆盖手册、说明类长尾问题；允许生成，但不替代 FAQ 的确定性 |

FAQ 不是唯一引擎，而是旁路式的确定性答案层：优先用 FAQ 锁口径，FAQ 不够再用文档 RAG 补覆盖。

当前已具备 FAQ CRUD、一阶段检索（ExactContainment 占位）与置信度分流骨架，便于端到端联调；FAQ 改写与文档 RAG 仍为占位实现，向量 / BM25 / 自研 embedding 后续接入。

## 整体流程

```text
用户提问
    │
    ▼
┌───────────────────┐
│  1. FAQ 检索      │  ← 第一步：标准问 / 相似问匹配
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
4. **检索层可演进**：当前 FAQ 为内存 CRUD；向量 / BM25 / 自研 embedding 后续替换占位实现。

## 环境

使用 conda 虚拟环境 `faq-rag`（Python 3.12+）。

```bash
conda activate faq-rag
pip install -e .
```

## 启动

先激活环境，并确认命令来自 conda（若本机同时装了 pyenv，容易误用全局 `fastapi`）：

```bash
conda activate faq-rag
which python   # 应指向 .../miniconda3/envs/faq-rag/bin/python
which fastapi  # 应指向 .../miniconda3/envs/faq-rag/bin/fastapi
```

推荐启动方式（不依赖 `fastapi` CLI，更稳）：

```bash
# 方式一：直接运行（热重载）
python main.py

# 方式二：uvicorn
uvicorn faq_rag.main:app --reload --host 0.0.0.0 --port 8000

# 方式三：FastAPI CLI（需 conda 环境内的 fastapi[standard]）
fastapi dev src/faq_rag/main.py
```

若 `fastapi` 报错 `please install "fastapi[standard]"`，多半是 pyenv shim 抢了命令，可任选其一：

```bash
# 看当前 fastapi 是否在 conda 环境内
which fastapi

# 用环境内绝对路径
"$CONDA_PREFIX/bin/fastapi" dev src/faq_rag/main.py

# 或先重装到当前环境，再确认 which
pip install -e .
hash -r
which fastapi
```

启动后访问：

- API 根路径：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- 提问入口：http://127.0.0.1:8000/ask
- 交互文档：http://127.0.0.1:8000/docs

## FAQ CRUD（当前为内存存储）

用于快速验证整体流程，进程重启后数据会丢失。

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

`POST /ask`：先 FAQ 检索，再按置信度分流。改写与文档 RAG 目前返回占位文案，响应里的 `route` / `notes` 可用于确认走了哪条分支。

| 字段 | 说明 |
|------|------|
| `question` | 用户提问 |
| `answer` | 最终答案（或占位文案） |
| `route` | `faq_verbatim` / `faq_rewrite` / `doc_rag` |
| `confidence` | FAQ 命中分；未命中时可能为 `null` |
| `faq_match` | 最佳 FAQ 命中详情（含 `matched_text`） |
| `notes` | 骨架阶段分流说明 |

示例（需先创建 FAQ）：

```bash
curl -s http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{
  "question": "Ollama 怎么装"
}'
```

当前占位阈值：≥ 0.90 → 原样返回；≥ 0.70 → 改写占位；否则 → 文档 RAG 占位。

## 项目结构

```text
main.py                        # 根入口（委托 faq_rag.main）
scripts/                       # 文档采集等脚本（见 scripts/README.md）
data/raw/ollama_docs/          # 采集得到的原始文档语料
src/faq_rag/
  main.py                      # FastAPI 应用入口
  models/
    faq.py                     # FAQ Pydantic 模型
    ask.py                     # 问答请求 / 响应模型
  services/
    faq/                       # FAQ 存储与检索
      store.py
      retriever.py
    similarity/                # 相似度索引（可替换实现）
      index.py
    ask/                       # 问答流水线与兜底
      pipeline.py
      rewriter.py
      doc_rag.py
  api/
    routes/
      health.py                # 健康检查
      faqs.py                  # FAQ CRUD
      ask.py                   # 提问入口
```

## 当前进度

- [x] FAQ CRUD（内存，流程验证）
- [x] FAQ 旁路门控骨架（极高置信原样 / 高置信改写占位 / 否则文档 RAG 占位）
- [x] 相似度索引骨架（ExactContainment；向量 / BM25 待接入）
- [ ] FAQ 改写（LLM，事实锁定）
- [ ] 向量与检索层（自研 embedding、BM25 等）
- [ ] 文档 RAG 粗粒度兜底
