"""Quick latency profiler — instruments each stage of the /answer pipeline.

Usage:
    python scripts/profile_answer_latency.py --project-id 1

Requires the FastAPI server to be running (uvicorn main:app).
"""
import argparse, asyncio, time, json, httpx, sys

BASE = "http://127.0.0.1:8010"
QUERY = "What is the preferred controller at Step 1 of asthma treatment?"


async def main(project_id: int):
    timings = {}
    t0 = time.perf_counter()

    # --- Stage 1: embed query ---
    # We mimic the query embedding via the embedding provider directly.
    # Instead, we call the search endpoint which does embed + search + rerank,
    # and the answer endpoint which does embed + search + rerank + generate + verify.
    # To isolate, we hit /search (no generation) and /answer (with generation).

    # /search covers: expand_query + embed + hybrid_search + rerank
    t_search_start = time.perf_counter()
    search_resp = await httpx_post(f"/api/v1/nlp/index/search/{project_id}",
                                   {"text": QUERY, "limit": 8, "retrieval_mode": "hybrid", "rerank": True, "score_threshold": None})
    timings["search_total (embed+hybrid+rerank)"] = time.perf_counter() - t_search_start

    # /answer covers: expand_query + embed + hybrid_search + rerank + generate + verify
    t_answer_start = time.perf_counter()
    answer_resp = await httpx_post(f"/api/v1/nlp/index/answer/{project_id}",
                                   {"text": QUERY, "limit": 8, "retrieval_mode": "hybrid", "rerank": True, "verify_claims": False})
    timings["answer_total (embed+search+rerank+generate)"] = time.perf_counter() - t_answer_start

    t1 = time.perf_counter()
    answer_text = answer_resp.get("answer", "")
    gen_allowed = answer_resp.get("confidence", {}).get("generation_allowed")
    timings["answer_length_chars"] = len(answer_text)
    timings["generation_allowed"] = gen_allowed

    print(json.dumps(timings, indent=2))


async def httpx_post(path: str, payload: dict):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{BASE}{path}", json=payload)
        return r.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main(args.project_id))
