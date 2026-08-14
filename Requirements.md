# AI Hackathon — Clinical RAG System

**Event:** AI Hackathons
**Organized by:** CREATIVA Innovation Hubs, ITIDA, TIEC, Orange Digital Center, INSTANT Software Solutions
**Dates:** August 16–20, 2026 (5 Days)
**Core Theme:** Build a safe, evidence-grounded, traceable, and trustworthy Retrieval-Augmented Generation (RAG) system using official medical guidelines.

> **Core Philosophy:** Clinical decision support must be grounded in official evidence with explicit citations, transparent retrieval, and verified refusal logic. Fluent ≠ Safe.
> 

---

## 📌 Project Scope

- [ ]  Pick **one narrow clinical topic** (e.g., Adult Hypertension Management, Diabetes Screening, or Asthma Guidance)
- [ ]  Source content **strictly** from official, public guidelines: WHO, CDC, NICE, USPSTF
- [ ]  No private or credential-gated data
- [ ]  Every clinical recommendation must trace back to a citable source

---

## Day 1 — Sun, Aug 16 (Offline)

**Focus:** Research, Scope & Ingestion, Team Formation
**Key Output:** Searchable Vector DB with Metadata

- [ ]  Form teams and assign roles
- [ ]  Finalize clinical topic scope
- [ ]  Source official guideline PDFs (WHO / CDC / NICE / USPSTF only)
- [ ]  Extract text from PDFs, preserving page numbers
- [ ]  Strip headers, footers, and extraction artifacts
- [ ]  Perform section-aware chunking (400–800 token chunks, preserving recommendation context and section boundaries)
- [ ]  Define metadata schema per chunk:
    - [ ]  `document_name`
    - [ ]  `page_number`
    - [ ]  `section_title`
    - [ ]  `chunk_id`
    - [ ]  `source_url`
- [ ]  Generate embeddings and load into vector DB
- [ ]  ✅ **Deliverable:** Searchable vector DB with full metadata

---

## Day 2 — Mon, Aug 17 (Offline)

**Focus:** Retrieval Optimization
**Key Output:** Measured Baseline & Precision@K

- [ ]  Build mini evaluation set of 15–20 labeled clinical test questions
- [ ]  Test Top-K tuning:
    - [ ]  Top-3 (precision-focused)
    - [ ]  Top-5 (balanced)
    - [ ]  Top-10 (recall-focused)
- [ ]  Test chunk size & overlap variants (400–600 vs. 700–900 tokens)
- [ ]  Compare retrieval strategies:
    - [ ]  Semantic search
    - [ ]  Keyword (BM25)
    - [ ]  Hybrid
    - [ ]  Reranking
- [ ]  Measure Precision@3 and Precision@5
- [ ]  Document failure modes and fixes
- [ ]  Build Evidence Panel UI (shows retrieved chunks, scores, metadata)
- [ ]  **Deliverable:** Measured baseline with Precision@K results

---

## Day 3 — Tue, Aug 18 (Online)

**Focus:** Grounded Generation & Citation
**Key Output:** Structured, Cited RAG Pipeline

- [ ]  Design generation prompt so retrieved guideline text is treated as absolute source of truth
- [ ]  Ensure LLM acts strictly as an **evidence synthesizer**, never a diagnostician
- [ ]  Implement structured output with inline citations (page/section level)
- [ ]  Bind every generated claim to its supporting chunk
- [ ]  Test end-to-end pipeline: query → retrieval → grounded generation → citation
- [ ]  **Deliverable:** Working structured, cited RAG pipeline

---

## Day 4 — Wed, Aug 19 (Online)

**Focus:** Safety, Guardrails & Internal Evaluation
**Key Output:** Guardrail Workflow & Benchmark Dashboard

- [ ]  Implement Input Risk Classification:
    - [ ]  Allowed
    - [ ]  Needs Caution (patient scenarios)
    - [ ]  Refuse/Redirect (emergencies, out-of-scope)
- [ ]  Implement Retrieval Confidence Thresholds (block/downgrade generation when similarity scores fall below cutoff)
- [ ]  Implement Unsupported Claim Detection (post-process outputs, verify every claim against retrieved chunk text)
- [ ]  Run benchmark across 20–30 structured test cases:
    - [ ]  Direct queries
    - [ ]  Multi-chunk queries
    - [ ]  Ambiguous queries
    - [ ]  Out-of-scope queries
- [ ]  Calculate metrics:
    - [ ]  Retrieval Precision@K = Relevant Chunks in Top-K
    - [ ]  Citation Faithfulness = Correct Supporting Citations / Total Citations Checked
    - [ ]  Unsupported Claim Rate = Total Unsupported Claims / Total Claims Generated
- [ ]  Build benchmark dashboard
- [ ]  **Deliverable:** Guardrail workflow + benchmark dashboard

---

## Day 5 — Thu, Aug 20 (Offline)

**Focus:** Final Presentation & Judge Evaluation
**Key Output:** Frozen Prototype, Live Demo & Pitches

- [ ]  Freeze prototype (no more changes)
- [ ]  Prepare clinical safety disclaimer ("supports — never replaces — clinical judgment")
- [ ]  Prepare scalability roadmap (single topic → multi-guideline, continuously validated KB)
- [ ]  Rehearse live demo script (see below)
- [ ]  Prepare final pitch deck
- [ ]  Present to judges
- [ ]  **Deliverable:** Frozen prototype, live demo, pitch

---

## System Architecture Checklist

Modular, layered pipeline — each stage independently testable and auditable.

- [ ]  **01 Ingestion** — official PDF sourcing + cleaning
- [ ]  **02 Chunking** — section-aware, metadata-tagged
- [ ]  **03 Embeddings** — vector generation
- [ ]  **04 Retrieval** — semantic / hybrid / reranked search
- [ ]  **05 Guardrails** — risk classification, confidence thresholds, claim verification
- [ ]  **06 Grounded LLM** — citation-bound generation
- [ ]  **07 Evidence Panel** — UI showing retrieved chunks, scores, metadata

---

## Live Demo Script (3 Cases)

- [ ]  **Case A — Success:** Single/multi-chunk query answered with exact page-level citations
- [ ]  **Case B — Complex Multi-Step:** Multi-section synthesis returned with fully structured supporting evidence
- [ ]  **Case C — Safe Refusal:** Out-of-scope query correctly triggers an insufficient-evidence refusal

---

## Judging Rubric

| Category | Weight | Key Evaluation Points |
| --- | --- | --- |
| Retrieval Quality | 30% | Relevant chunk extraction, Precision@K scores, chunk metadata transparency, vector index structure |
| Grounding & Citation | 25% | Accurate page/section citations, zero unsupported claims, minimal hallucination, tight claim-to-chunk binding |
| System Architecture | 15% | Modular pipeline design, clear separation of layers, clean architectural diagram |
| Evaluation Depth | 15% | ≥2 quantitative metrics reported (20+ Q dataset), clear failure analysis and trade-off understanding |
| Safety & UX | 15% | Input guardrails, safe refusal logic, confidence indicators, clinical disclaimer, smooth live presentation & demo |

---

## Pre-Submission Checklist

- [ ]  Topic scoped to one clinical area, sourced only from WHO/CDC/NICE/USPSTF
- [ ]  Vector DB with full metadata schema
- [ ]  Precision@K measured and documented
- [ ]  Citations traceable to page/section level
- [ ]  Guardrails: risk classification, confidence threshold, unsupported claim detection
- [ ]  ≥20 test case benchmark with ≥2 quantitative metrics
- [ ]  Clinical disclaimer included in UI
- [ ]  All 3 demo cases rehearsed (success, complex, refusal)
- [ ]  Architectural diagram finalized
- [ ]  Scalability roadmap prepared