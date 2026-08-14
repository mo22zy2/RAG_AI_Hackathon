"""
Retrieval evaluation harness.

Runs the golden-labeled questions from eval/clinical_questions_v2.json against a
running server and computes Precision@K and HitRate@K for every retrieval
mode x Top-K combination.

True-negative questions (expect_hit=false, e.g. q21/q22) are scored separately:
instead of P@K/HitRate they are checked against RETRIEVAL_SCORE_THRESHOLD —
the top retrieved chunk's similarity score must fall below the threshold (or no
results returned) for the pipeline to behave correctly. They are excluded from
the standard P@K/HitRate means so out-of-corpus questions don't pollute them.

Prerequisites:
    - The app is running (uvicorn main:app --reload).
    - A project has been uploaded, processed and indexed
      (/api/v1/data/upload, /api/v1/data/process, /api/v1/nlp/index/push).

Usage (run from src/):
    python ../scripts/eval_retrieval.py --project-id 1
    python ../scripts/eval_retrieval.py --project-id 1 --modes vector hybrid rerank --top-k 3 5
    python ../scripts/eval_retrieval.py --project-id 1 --base-url http://localhost:8000
    python ../scripts/eval_retrieval.py --project-id 1 --tn-threshold 0.35

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

# True-negative scoring: a question with expect_hit=false is handled correctly
# only when the best result's score stays below this threshold (or no results
# are returned at all). Mirrors RETRIEVAL_SCORE_THRESHOLD in src/.env.
TN_THRESHOLD = 0.35
# Only these modes return calibrated similarity/relevance scores in [0, 1].
# keyword (ts_rank) and hybrid (RRF) scores are rank-fusion values and cannot
# be compared against a similarity threshold, so they are reported but not
# scored.
CALIBRATED_MODES = ("vector", "rerank")


def is_true_negative(q):
    return q.get("expect_hit") is False or not q.get("golden")


def top_score_of(results):
    scores = []
    for c in results or []:
        s = c.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
    return max(scores) if scores else None


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
        is_tn = is_true_negative(q)
        ks = [1] if is_tn else top_ks
        for mode in modes:
            for k in ks:
                payload = {
                    "text": q["question"],
                    "limit": k,
                    "retrieval_mode": mode,
                    "rerank": True if mode == "rerank" else False,
                }
                try:
                    body = call_search(base_url, project_id, payload)
                    results = body.get("results") or []
                except Exception as e:
                    row = {
                        "qid": q["id"], "mode": mode, "k": k,
                        "precision": None, "hit": None, "error": str(e), "results": [],
                        "is_tn": is_tn, "top_score": None, "tn_ok": None,
                    }
                    rows.append(row)
                    continue

                if is_tn:
                    top_score = top_score_of(results)
                    tn_ok = None
                    if mode in CALIBRATED_MODES:
                        tn_ok = (top_score is None) or (top_score < TN_THRESHOLD)
                    rows.append({
                        "qid": q["id"], "mode": mode, "k": k,
                        "precision": None, "hit": None, "error": None,
                        "results": [c.get("chunk_id") for c in results],
                        "is_tn": True, "top_score": top_score, "tn_ok": tn_ok,
                    })
                else:
                    relevant = [is_relevant(c, q["golden"]) for c in results]
                    precision = (sum(relevant) / k) if results else 0.0
                    hit = 1 if any(relevant) else 0
                    rows.append({
                        "qid": q["id"], "mode": mode, "k": k,
                        "precision": precision, "hit": hit, "error": None,
                        "results": [c.get("chunk_id") for c in results],
                        "is_tn": False, "top_score": None, "tn_ok": None,
                    })
    return rows


def build_markdown(questions, rows):
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    in_scope = [q for q in questions if not is_true_negative(q)]
    tn_questions = [q for q in questions if is_true_negative(q)]
    in_scope_rows = [r for r in rows if not r.get("is_tn")]
    tn_rows = [r for r in rows if r.get("is_tn")]

    lines.append("# Retrieval Evaluation Results")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Questions:** {len(questions)} total "
                 f"({len(in_scope)} in-scope, {len(tn_questions)} true-negative) | "
                 f"**Modes:** {MODES} | **Top-K:** {TOP_KS}")
    lines.append("**Relevance rule:** golden doc hint in `document_name` (case-insensitive) AND "
                 f"`page_number` within +/-{PAGE_WINDOW} of the golden page window AND any golden "
                 "keyword present in the chunk text.")
    lines.append(f"**True-negative rule:** `expect_hit=false` questions are excluded from "
                 f"P@K/HitRate means. For `{', '.join(CALIBRATED_MODES)}` modes they pass only if "
                 f"no result is returned or the top similarity score < {TN_THRESHOLD}. "
                 "`keyword`/`hybrid` scores are rank-fusion values and are reported but not scored.")
    lines.append("")

    # Per-question table (in-scope only)
    lines.append("## Per-question (in-scope)")
    header = "| qid | mode | K | P@K | Hit | error |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for r in in_scope_rows:
        p = f"{r['precision']:.2f}" if r["precision"] is not None else "-"
        h = "yes" if r["hit"] else "no" if r["hit"] is not None else "-"
        err = r["error"] or ""
        lines.append(f"| {r['qid']} | {r['mode']} | {r['k']} | {p} | {h} | {err} |")
    lines.append("")

    # Summary (in-scope only)
    lines.append("## Summary (in-scope questions, mean over questions)")
    lines.append("| mode | K | mean P@K | hit rate |")
    lines.append("|---|---|---|---|")
    valid = [r for r in in_scope_rows if r["precision"] is not None]
    for mode in MODES:
        for k in TOP_KS:
            subset = [r for r in valid if r["mode"] == mode and r["k"] == k]
            if not subset:
                continue
            mean_p = sum(r["precision"] for r in subset) / len(subset)
            hit_rate = sum(r["hit"] for r in subset) / len(subset)
            lines.append(f"| {mode} | {k} | {mean_p:.3f} | {hit_rate:.3f} |")
    lines.append("")

    # True-negative behavior
    lines.append("## True-negative behavior (expect_hit=false, top score check)")
    lines.append(f"Threshold: top similarity score < {TN_THRESHOLD} (or empty) = correct. "
                 "Only `vector`/`rerank` are scored; `keyword`/`hybrid` are informational.")
    lines.append("| qid | mode | top score | result |")
    lines.append("|---|---|---|---|")
    for r in tn_rows:
        if r.get("error"):
            lines.append(f"| {r['qid']} | {r['mode']} | - | error: {r['error']} |")
            continue
        score = f"{r['top_score']:.4f}" if r["top_score"] is not None else "empty"
        if r["tn_ok"] is True:
            result = "ok"
        elif r["tn_ok"] is False:
            result = "SPURIOUS MATCH"
        else:
            result = "n/a (uncalibrated score)"
        lines.append(f"| {r['qid']} | {r['mode']} | {score} | {result} |")

    scored_tn = [r for r in tn_rows if r["tn_ok"] is not None]
    if scored_tn:
        tn_ok_count = sum(1 for r in scored_tn if r["tn_ok"])
        lines.append("")
        lines.append(f"True-negative accuracy (scored modes only): "
                     f"**{tn_ok_count}/{len(scored_tn)}** "
                     f"({tn_ok_count / len(scored_tn):.2%})")
    lines.append("")

    # Failure modes (in-scope only)
    lines.append("## Failure modes (in-scope, no golden hit)")
    for r in valid:
        if r["hit"] == 0:
            q = next((q for q in in_scope if q["id"] == r["qid"]), None)
            title = q["question"][:90] + "..." if q and len(q["question"]) > 90 else (q["question"] if q else "")
            lines.append(f"- **{r['qid']}** ({r['mode']}, K={r['k']}): {title}")
    lines.append("")

    # Spurious true-negative retrievals
    spurious = [r for r in tn_rows if r["tn_ok"] is False]
    if spurious:
        lines.append("## Spurious true-negative retrievals (confident match on out-of-corpus question)")
        for r in spurious:
            q = next((q for q in tn_questions if q["id"] == r["qid"]), None)
            title = q["question"][:90] + "..." if q and len(q["question"]) > 90 else (q["question"] if q else "")
            lines.append(f"- **{r['qid']}** ({r['mode']}, top_score={r['top_score']:.3f}): {title}")
        lines.append("")

    return "\n".join(lines)


def main():
    global TN_THRESHOLD
    parser = argparse.ArgumentParser(description="Retrieval Precision@K evaluation")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--top-k", nargs="+", type=int, default=TOP_KS)
    parser.add_argument("--tn-threshold", type=float, default=TN_THRESHOLD,
                        help="Similarity threshold for expect_hit=false questions")
    args = parser.parse_args()

    TN_THRESHOLD = args.tn_threshold

    if not QUESTIONS_PATH.exists():
        print(f"Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    in_scope = sum(1 for q in questions if not is_true_negative(q))
    tn = len(questions) - in_scope
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH.name} "
          f"({in_scope} in-scope, {tn} true-negative)")
    print(f"Target: {args.base_url}, project_id={args.project_id}, "
          f"modes={args.modes}, top-k={args.top_k}, tn-threshold={args.tn_threshold}")

    start = time.time()
    rows = run_eval(args.base_url, args.project_id, questions, args.modes, args.top_k)
    elapsed = time.time() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text(build_markdown(questions, rows), encoding="utf-8")
    print(f"\nDone in {elapsed:.1f}s -> {out_path}")

    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"WARNING: {len(errors)} failed requests (e.g. {errors[0]['error'][:120]}). "
              "Is the server running with the project indexed?")


if __name__ == "__main__":
    main()
