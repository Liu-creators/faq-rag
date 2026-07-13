# 后续优化：HyDE / 多查询改写

> 状态：思路记录，尚未实现。  
> 相关代码：`src/faq_rag/services/doc/retriever.py`、`src/faq_rag/services/ask/doc_rag.py`、`src/faq_rag/services/llm/client.py`

## 现状

文档 RAG 与 FAQ 检索都是**用户原句直接检索**：

```text
用户问题（口语、短、省略）
  → embed(query) + BM25(query)
  → Milvus 混合召回
```

知识库里的切块往往是说明书口吻（完整句、术语、标题路径），和用户问法存在 **query–document 表述鸿沟**：语义相近但字面差大时，dense / BM25 都容易漏召或排错。

一期未做查询侧改写，是为了先稳定「入库切块 + 检索 + 生成」主链路。

## 核心想法

在检索**之前**改写或扩展查询，让检索输入更接近语料写法，或覆盖多种问法。两条常见路，别混成一件事：

| 方法 | 在做什么 | 典型收益 |
|------|----------|----------|
| **多查询改写** | 一问变多问（同义、拆分、中英、补全），分别检索再合并 | 召回更全，漏检少 |
| **HyDE** | 让 LLM 先写一段「假想答案/假想文档」，再对假文档做向量检索 | 短问、口语问更贴近文档向量空间 |

二者都只改**召回输入**，不直接改最终答案；生成仍基于真实切块，并遵守现有事实锁定约束。

## 多查询改写（Multi-query）

### 流程

```text
用户问题
  → LLM / 规则 生成 Q1…Qn（n 通常 3～5）
  → 对每个 Qi 做 Milvus search
  → 按 doc_id（chunk id）合并、去重
  → 用 max(score) 或 RRF 跨查询融合排序
  → 截断 top_k → 生成（或再接重排）
```

### 示例

用户：「ollama 怎么装到 Mac 上」

可能扩展为：

1. macOS 安装 Ollama 的步骤  
2. How to install Ollama on Mac  
3. Ollama brew 安装  

### 本项目落点

- 入口：`answer_from_documents` 或 `DocChunkRetriever.retrieve` 前增加 `expand_queries(question) -> list[str]`。
- 融合：复用「多路命中同一 `chunk.id` 取最高分」；多路查询之间可用简易 RRF（与 Milvus 内 dense/BM25 的 RRF 类似，但是跨 query）。
- 成本：检索次数 ≈ n；可用较小 LLM / 低 temperature，或对高频问缓存改写结果。

## HyDE（Hypothetical Document Embeddings）

### 流程

```text
用户问题
  → LLM 生成「假想文档段落」（像手册里会写的说明，而非最终对用户的客服话术）
  → embed(假想文档) 做 dense 检索
  → （可选）仍用原问题做 BM25，再与 dense 结果融合
  → 取真实切块 → 生成
```

要点：假想文档**只用于检索**，应答内容必须来自 MySQL / Milvus 命中的真实 `doc_chunks`。

### 为何有效

用户问句短、向量偏「问句空间」；假想答案更接近「文档空间」，和切块 embedding 更同分布，ANN 更易命中。

### 风险

- 假想内容若胡编，可能检索到错误主题的块 → 假想 prompt 要约束「简短、中性、像文档摘录」，并限制长度。
- 多一次 LLM 调用，延迟上升；失败时应**回退原句检索**。
- 单独 HyDE 偏 dense；本项目已有 BM25，建议 **HyDE dense + 原问 BM25** 混合，避免关键词完全丢掉。

## 建议路线（由轻到重）

1. **规则 / 轻量改写**  
   去语气词、补产品名、中英同现；无 LLM 或极少调用。先验证表述鸿沟是否真是瓶颈。

2. **多查询改写（优先于 HyDE）**  
   实现简单、可解释；与现有 `search` 组合清晰。适合中文手册 + 口语问。

3. **HyDE**  
   短问、召回仍差时再加；与多查询可并存（对每个改写问做 HyDE 成本过高，一般二选一或「原问 HyDE + 1～2 个改写问」）。

4. **再接 Cross-encoder 重排**  
   查询扩展提高召回后，候选变多、噪声也变多，适合接 [`cross-encoder-rerank.md`](cross-encoder-rerank.md) 做精排。

## 配置设想（尚未落地）

| 变量 | 含义 |
|------|------|
| `DOC_QUERY_EXPAND_ENABLED` | 是否启用多查询 |
| `DOC_QUERY_EXPAND_N` | 扩展问句数 |
| `DOC_HYDE_ENABLED` | 是否启用 HyDE |
| `DOC_HYDE_MODEL` | 可选，默认跟 `LLM_MODEL` |

FAQ 路径也可对相似问扩展做同样事情，但 FAQ 已有人工 `similar_questions`，收益通常小于文档长尾。

## 与生成层的边界

| 阶段 | 允许 | 不允许 |
|------|------|--------|
| 查询改写 / HyDE | 改写问法、写假想文档供检索 | 把假想文档当答案返回 |
| 生成（现有 `generator.py`） | 只依据真实检索片段作答 | 用改写模型的臆测补事实 |

## 小结

- **多查询**：一问多路检索，主打召回覆盖。  
- **HyDE**：用假想文档向量检索，主打缩小问句与文档的表述差。  
- 本项目更自然的顺序：先多查询 + 融合 → 不够再上 HyDE → 候选变多后上 cross-encoder 重排。
