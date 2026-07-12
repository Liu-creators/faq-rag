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
