# Scripts

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
