# 后续优化：Cross-encoder 重排

> 状态：思路记录，尚未实现。  
> 相关代码：`src/faq_rag/services/doc/retriever.py`、`src/faq_rag/services/ask/doc_rag.py`、`src/faq_rag/services/similarity/milvus_index.py`

## 现状

文档 RAG 当前是**单阶段检索**：

```text
用户问题
  → Milvus 混合召回（dense ANN + BM25 / RRF）
  → 稠密余弦打分排序
  → 取 top_k（默认 DOC_RAG_TOP_K=5）
  → 直接拼进 prompt 生成
```

召回与排序共用同一套向量 / 关键词信号。向量检索快、可扩，但对「问句与切块是否真相关」的细粒度判断偏弱：容易把字面相近、主题不对的块排到前面，挤占生成上下文。

## 核心想法

把流水线拆成两段：

1. **召回（Recall）**：继续用现有 Milvus 混合检索，多取一些候选（如 top 20～50）。
2. **重排（Rerank）**：用 **Cross-encoder** 对「问题 + 单条切块」联合打分，按新分数重排，再截断到生成用的 top_k（如 3～5）。

| | Bi-encoder（当前 dense） | Cross-encoder |
|--|--------------------------|---------------|
| 编码方式 | 问题、文档各自编码，再比向量 | 问题与文档拼在一起，一次前向出分 |
| 速度 | 快，适合全库 ANN | 慢，只适合少量候选 |
| 相关性 | 粗粒度语义 | 通常更准（词对齐、否定、细节） |

Cross-encoder **不替代** Milvus，而是挂在召回之后。

## 建议落点（本项目）

```text
DocChunkRetriever.retrieve
  → index.search(top_k=recall_k)          # 已有
  → cross_encoder.rerank(question, hits)  # 新增
  → 截断为 DOC_RAG_TOP_K
  → answer_from_documents / generator
```

可选实现形态：

1. **本地 sentence-transformers cross-encoder**（如 `ms-marco-MiniLM` / 中文对等模型）  
   - 进程内推理，延迟可控；需选中文或中英双语模型。
2. **OpenAI 兼容的 rerank API**（若所用网关提供）  
   - 与现有 embedding / LLM 配置风格一致，运维简单。

配置设想（尚未落地）：

| 变量 | 含义 |
|------|------|
| `DOC_RAG_RECALL_K` | 重排前召回条数（如 30） |
| `DOC_RERANK_ENABLED` | 是否启用重排 |
| `DOC_RERANK_MODEL` | 模型名或路径 |

对外置信度 / `sources[].score`：重排后应用 **rerank 分**（或归一化后的分）展示；Milvus 余弦可保留为调试字段。

## 与 FAQ 路径的关系

- **文档 RAG**：优先受益；切块多、问法杂，重排收益通常更大。
- **FAQ 检索**：也可对 top_n FAQ 候选做 cross-encoder 再分流；与 [`learned-confidence-routing.md`](learned-confidence-routing.md) 中的「检索后打分 / 路由头」是同一类能力，可共用模型，阈值不同。

一期文档 RAG 未做重排，是为了先跑通入库 → 检索 → 生成。

## 风险与注意

- **延迟**：候选数 × 模型耗时；需限制 `recall_k`，必要时批量推理。
- **分数不可比**：cross-encoder 原始分通常不是 0～1 概率，不要直接套 FAQ 的 `0.90` / `0.70`。
- **语言与领域**：通用英文重排模型对中文手册可能偏弱，需评测后再定模型。
- **无命中**：重排不能从零变出相关文档；召回本身差时要先调切块 / 混合检索 / 查询改写（见 [`query-expansion-hyde.md`](query-expansion-hyde.md)）。

## 小结

- 短期：保持现状；用日志看「生成用了错误切片」是否频繁。
- 中期：召回放大 + 本地 / API cross-encoder，只改 `DocChunkRetriever` 出口。
- 长期：与可学习分流共用「问题–候选」打分特征，FAQ 与文档两条路径统一校准。
