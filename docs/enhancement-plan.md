# RAG Enhancement Plan

## Phase 1 — Foundation Fixes ✅

| # | Enhancement | Status |
|---|------------|--------|
| 1 | Use `RecursiveCharacterTextSplitter` instead of custom `\n`-splitter | Skipped (custom splitter kept) |
| 2 | **Fix `chunk_order` to Integer** — column type changed from `String` to `Integer`. Migration: `a1b2c3d4e5f6` | ✅ Done |
| 3 | **Store real metadata** — each chunk now stores `asset_id`, `file_name`, `project_id`, `chunk_order` in `chunk_metadata` | ✅ Done |
| 4 | **Add EUCLIDEAN distance** — `VECTOR_DB_DISTANCE_METHOD="euclidean"` now works with `<->` operator and `vector_l2_ops` index | ✅ Done |

## Phase 2 — Retrieval Quality

| # | Enhancement | Why |
|---|------------|-----|
| 5 | **Hybrid search** — combine vector similarity with PostgreSQL `tsvector` full-text search | Catches exact keyword matches (IDs, names, numbers) that pure vector search misses |
| 6 | **Cross-encoder reranking** (e.g., Cohere Rerank API) | First-stage retrieval returns top-K, reranker re-scores for precision |
| 7 | **Semantic chunking** — split on paragraph/sentence boundaries via `RecursiveCharacterTextSplitter` | Keeps coherent ideas in one chunk |
| 8 | **Query rewriting** — expand short user queries before embedding | Short queries produce weak vectors; rewriting boosts recall |

## Phase 3 — UX & Observability

| # | Enhancement | Why |
|---|------------|-----|
| 9 | **Streaming responses** from the answer endpoint | Users see tokens arrive live instead of waiting 10+ seconds |
| 10 | **Source citations** — return which chunks were used | Builds trust and lets users verify answers |
| 11 | **Language auto-detection** → switch templates + embedding model | Currently hardcoded `DEFAULT_LANGUAGE`; auto-detect Arabic and load `ar/rag.py` |
| 12 | **Request tracing / structured logging** | Impossible to debug a failed RAG query end-to-end today |

## Phase 4 — Advanced

| # | Enhancement | Why |
|---|------------|-----|
| 13 | **Multi-turn conversation** — pass previous Q&A as context | Every query is currently independent |
| 14 | **Support more file types** (DOCX, HTML, CSV, Markdown) | Currently only TXT and PDF |
| 15 | **Embedding cache** — avoid re-embedding identical chunks on re-index | Saves API costs and time |
| 16 | **Chunk summary indexing** — embed a summary instead of the raw chunk | Long chunks dilute the embedding signal |
| 17 | **Prompt A/B testing** — evaluate different system prompts | One hardcoded prompt per locale today |
