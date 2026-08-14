"""
Retrieval evaluation harness.

Runs the golden-labeled questions from eval/clinical_questions_v2.json against a
running server and computes Precision@K and HitRate@K for every retrieval
mode x Top-K combination.

True-negative questions (expect_hit=false, e.g. q21/q22) are excluded from the
precision/hit-rate means and reported as a rejection rate instead.

Prerequisites:
    - The app is running (uvicorn main:app --reload).
    - A project has been uploaded, processed and indexed
      (/api/v1/data/upload, /api/v1/data/process, /api/v1/nlp/index/push).

Usage (run from src/):
    python ../scripts/eval_retrieval.py --project-id 1
    python ../scripts/eval_retrieval.py --project-id 1 --modes vector hybrid rerank --top-k 3 5
    python ../scripts/eval_retrieval.py --project-id 1 --base-url http://localhost:8000

Output:
    eval/results_<timestamp>.md
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "clinical_questions_v2.json"
OUTPUT_DIR = REPO_ROOT / "eval"

MODES = ["vector", "keyword", "hybrid", "rerank"]
TOP_KS = [3, 5, 10]
PAGE_WINDOW = 2


def is_relevant(chunk, golden_entries):
    text = (chunk.get("text") or "").lower()
    metadata = chunk.get("metadata") or {}
    document_name = (metadata.get("document_name") or "").lower()
    page = metadata.get("page_number")

    for entry in golden_entries:
        if entry.get("doc_hint") and entry["doc_hint"].lower() not in document_name:
            continue
        if page is not None:
            if page < entry["page_from"] - PAGE_WINDOW or page > entry["page_to"] + PAGE_WINDOW:
                continue
        keywords = [k for k in entry.get("keywords", []) if k]
        if keywords and not any(k.lower() in text for k in keywords):
            continue
        return True
    return False


def call_search(base_url, project_id, payload):
    url = f"{base_url.rstrip('/')}/api/v1/nlp/index/search/{project_id}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_eval(base_url, project_id, questions, modes, top_ks):
    rows = []
    for q in questions:
        for mode in modes:
            for k in top_ks:
                payload = {
                    "text": q["question"],
                    "limit": k,
                    "retrieval_mode": mode,
                    "rerank": True if mode == "rerank" else False,
                    "expand_query": True,
                }
                try:
                    body = call_search(base_url, project_id, payload)
                    results = body.get("results") or []
                except Exception as e:
                    rows.append({
                        "qid": q["id"], "mode": mode, "k": k,
                        "precision": None, "hit": None, "error": str(e), "results": [],
                    })
                    continue

                relevant = [is_relevant(c, q["golden"]) for c in results]
                precision = (sum(relevant) / k) if results else 0.0
                hit = 1 if any(relevant) else 0
                rows.append({
                    "qid": q["id"], "mode": mode, "k": k,
                    "precision": precision, "hit": hit, "error": None,
                    "results": [c.get("chunk_id") for c in results],
                })
    return rows


def build_markdown(questions, rows):
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# Retrieval Evaluation Results")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Questions:** {len(questions)} | **Modes:** {MODES} | **Top-K:** {TOP_KS}")
    lines.append("**Retrieval:** query expansion enabled (mirrors the production /answer path).")
    lines.append("**True-negatives:** questions with `expect_hit=false` are excluded from precision/hit-rate means "
                 "and reported as a rejection rate instead.")
    lines.append("**Relevance rule:** golden doc hint in `document_name` (case-insensitive) AND "
                 f"`page_number` within +/-{PAGE_WINDOW} of the golden page window AND any golden "
                 "keyword present in the chunk text.")
    lines.append("")

    # Per-question table
    lines.append("## Per-question")
    header = "| qid | mode | K | P@K | Hit | error |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        p = f"{r['precision']:.2f}" if r["precision"] is not None else "-"
        h = "yes" if r["hit"] else "no" if r["hit"] is not None else "-"
        err = r["error"] or ""
        lines.append(f"| {r['qid']} | {r['mode']} | {r['k']} | {p} | {h} | {err} |")
    lines.append("")

    # Summary
    tn_ids = {q["id"] for q in questions if q.get("expect_hit") is False}
    lines.append("## Summary (mean over positive questions; true-negatives reported separately)")
    lines.append("| mode | K | mean P@K | hit rate |")
    lines.append("|---|---|---|---|")
    valid = [r for r in rows if r["precision"] is not None]
    positive = [r for r in valid if r["qid"] not in tn_ids]
    tn_rows = [r for r in valid if r["qid"] in tn_ids]
    for mode in MODES:
        for k in TOP_KS:
            subset = [r for r in positive if r["mode"] == mode and r["k"] == k]
            if not subset:
                continue
            mean_p = sum(r["precision"] for r in subset) / len(subset)
            hit_rate = sum(r["hit"] for r in subset) / len(subset)
            lines.append(f"| {mode} | {k} | {mean_p:.3f} | {hit_rate:.3f} |")
    lines.append("")
    tn_subset = [r for r in tn_rows if r["k"] == TOP_KS[0]]
    if tn_subset:
        tn_qids = sorted({r["qid"] for r in tn_subset})
        rejection = sum(1 for r in tn_subset if r["hit"] == 0) / len(tn_subset)
        lines.append(
            f"**True-negative rejection rate (K={TOP_KS[0]}):** {rejection:.3f} "
            f"({len(tn_qids)} true-negative questions - {', '.join(tn_qids)} - "
            "correctly returned no relevant chunks)."
        )
        lines.append("")

    # Failure modes
    lines.append("## Failure modes (positive queries, no golden hit)")
    for r in positive:
        if r["hit"] == 0:
            q = next((q for q in questions if q["id"] == r["qid"]), None)
            title = q["question"][:90] + "..." if q and len(q["question"]) > 90 else (q["question"] if q else "")
            lines.append(f"- **{r['qid']}** ({r['mode']}, K={r['k']}): {title}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Precision@K evaluation")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--top-k", nargs="+", type=int, default=TOP_KS)
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        print(f"Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH.name}")
    print(f"Target: {args.base_url}, project_id={args.project_id}, "
          f"modes={args.modes}, top-k={args.top_k}")

    start = time.time()
    rows = run_eval(args.base_url, args.project_id, questions, args.modes, args.top_k)
    elapsed = time.time() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f"results_{ts}.md"
    out_path.write_text(build_markdown(questions, rows), encoding="utf-8")
    rows_path = OUTPUT_DIR / f"retrieval_rows_{ts}.json"
    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nDone in {elapsed:.1f}s -> {out_path}")
    print(f"Rows -> {rows_path}")

    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"WARNING: {len(errors)} failed requests (e.g. {errors[0]['error'][:120]}). "
              "Is the server running with the project indexed?")


if __name__ == "__main__":
    main()
