# Scalability Roadmap
**Path to a Continuous Medical Knowledge Base**

While the current iteration of the Mini RAG system is focused on a narrow clinical scope (Asthma) for the hackathon, the system is architecturally designed to scale.

## Phase 1: Multi-Guideline Expansion (Months 1-3)
**Goal:** Expand from a single clinical topic to multiple interconnected domains (e.g., Asthma, Diabetes, Hypertension).

- **Multi-Tenant Collections:** The system already supports isolated `project_id` namespaces, resulting in separate vector collections (e.g., `collection_1024_1`). We can seamlessly index Diabetes guidelines under `project_id=2` without overlapping scopes.
- **Dynamic Routing:** Implement an LLM-based query router that inspects the query and directs the hybrid search to the appropriate vector collection, or fans out the search across multiple collections for comorbid conditions (e.g., Asthma + Hypertension).
- **Automated Guideline Ingestion:** Connect the ingestion pipeline directly to RSS feeds or APIs from WHO, NICE, and CDC to flag when guidelines are updated.

## Phase 2: Enhanced Retrieval & Safety (Months 3-6)
**Goal:** Achieve enterprise-grade clinical precision.

- **Knowledge Graphs (GraphRAG):** Supplement the vector database with a clinical Knowledge Graph (e.g., Neo4j). This allows the system to understand exact relationships (e.g., "Drug A *contraindicates* Condition B") that vector similarity sometimes misses.
- **Agentic Verification Pipeline:** Replace the current regex-based citation checker with a secondary small LLM (e.g., Llama 3 8B) dedicated entirely to verifying claims against the retrieved context before serving the response to the user.
- **Streaming Responses:** Implement Server-Sent Events (SSE) to stream the Evidence Panel first, followed by the LLM generation token-by-token, improving perceived latency for end users.

## Phase 3: Clinical System Integration (Months 6-12)
**Goal:** Move from a standalone web app to an integrated clinical decision support tool.

- **SMART on FHIR Integration:** Adapt the API to accept FHIR-formatted patient contexts safely, strictly for retrieving relevant guidelines (never sending PHI to external LLM providers).
- **Local Deployment (Air-gapped):** Transition generation from Cohere/OpenAI-compatible endpoints to purely local, air-gapped instances of medically tuned models (e.g., Meditron or specialized Llama 3 variants) running on on-premise hardware via Ollama or vLLM to satisfy strict hospital data governance. 
- **Continuous Evaluation:** Build an automated feedback loop where clinician "thumbs up/down" ratings on answers are logged and automatically added to the `clinical_questions.json` evaluation dataset to prevent regressions in future model updates.
