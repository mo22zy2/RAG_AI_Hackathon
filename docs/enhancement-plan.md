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
| 5 | **Hybrid search** — combine vector similarity with PostgreSQL `tsvector` full-text search | ✅ Done |
| 6 | **Cross-encoder reranking** (e.g., Cohere Rerank API) | ✅ Done — model was `rerank-v3.0` (deprecated, 404 → silently fell back to hybrid). Switched to `rerank-v3.5`. Now genuinely re-orders; rerank@3 hit rate jumped 0.60 → 0.90 |
| 7 | **Semantic chunking** — split on paragraph/sentence boundaries via `RecursiveCharacterTextSplitter` | ✅ Done — fixed heading detection + tiny-fragment merge in the custom splitter (NHLBI QRG: 1,048 → 613 chunks) |
| 8 | **Query rewriting** — expand short user queries before embedding | Open |

## Eval results (baseline, 20 clinical questions, golden page windows)

Hit rate @10 / mean P@K — `eval/results_20260811_200837.md`

| mode | @3 hit | @5 hit | @10 hit | @3 P | @10 P |
|---|---|---|---|---|---|
| rerank | **0.90** | **0.95** | 0.95 | **0.367** | 0.165 |
| hybrid | 0.60 | 0.80 | **1.00** | 0.217 | 0.155 |
| vector | 0.65 | 0.75 | 0.80 | 0.217 | 0.135 |
| keyword | 0.50 | 0.70 | 0.80 | 0.167 | 0.135 |

- `q16` (NHLBI step tables) went 0/12 → 12/12 after the chunker fix.
- Best mode at small K is **rerank**; **hybrid** is the only mode that hits 1.00 @10.
- Remaining weak spots: `q11` (obesity, keyword/rerank miss — rerank candidate pool is keyword-based), `q08` (MART, vector miss), `q02` (vector-only miss).

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
| 18 | **Medical safety guardrail** — detect queries about a specific person's symptoms; force a refusal + clinician-referral prefix before general info | ✅ Done — rule-based detector in `NLPController` (Case C demo works) |
