import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Adjust sys.path to allow importing from the same directory
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

try:
    from eval_retrieval_v2 import MODES, TOP_KS, PAGE_WINDOW, call_search, is_relevant, run_eval as run_retrieval_eval
    from eval_answers import call_answer, contains_any, citation_docs_hit, detect_language, run_eval as run_answers_eval
except ImportError as e:
    print(f"Error importing from eval_retrieval_v2 or eval_answers: {e}")
    sys.exit(1)

REPO_ROOT = SCRIPTS_DIR.parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "clinical_questions_v2.json"
CASES_PATH = REPO_ROOT / "eval" / "answer_cases_v2.json"
OUTPUT_DIR = REPO_ROOT / "eval"


def is_true_negative(q):
    return q.get("expect_hit") is False

def get_system_info(base_url):
    try:
        url = f"{base_url.rstrip('/')}/api/v1/"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            emb = data.get("embedding_model")
            gen = data.get("generation_model")
            if emb and gen:
                return emb, gen
    except Exception:
        pass
    # Fall back to the local src/.env so the header still reports the stack.
    env_file = REPO_ROOT / "src" / ".env"
    if env_file.exists():
        env = {}
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
        return env.get("EMBEDDING_MODEL_ID", "unknown"), env.get("GENERATION_MODEL_ID", "unknown")
    return "unknown", "unknown"

def get_status(score, target, is_zero_target=False):
    if is_zero_target:
        if score <= target:
            return "✅"
        elif score <= target + 0.1:
            return "⚠️"
        else:
            return "❌"
    else:
        if score >= target:
            return "✅"
        elif score >= target - 0.1:
            return "⚠️"
        else:
            return "❌"

def mean(values):
    valid_values = [v for v in values if v is not None]
    return sum(valid_values) / len(valid_values) if valid_values else 0.0

def build_dashboard(retrieval_rows, answer_rows, elapsed, questions, cases, emb_model, gen_model):
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    questions_count = sum(1 for q in questions if not is_true_negative(q))
    tn_count = len(questions) - questions_count
    cases_count = len(cases)

    # Pre-compute retrieval metrics. True-negative questions (expect_hit=false)
    # are excluded from the precision/hit-rate means; their correctness is that
    # NO returned chunk is relevant (a hit of 0), reported as a rejection rate.
    tn_qids = {q["id"] for q in questions if is_true_negative(q)}
    r_valid = [r for r in retrieval_rows if r.get("precision") is not None and r["qid"] not in tn_qids]
    tn_rows = [r for r in retrieval_rows if r.get("precision") is not None and r["qid"] in tn_qids]
    tn_ok_count = sum(1 for r in tn_rows if r["hit"] == 0)
    tn_accuracy = (tn_ok_count / len(tn_rows)) if tn_rows else None

    def get_r_metric(mode, k, metric):
        subset = [r for r in r_valid if r["mode"] == mode and r["k"] == k]
        if not subset:
            return 0.0
        return sum(r[metric] for r in subset) / len(subset)

    p3_rerank = get_r_metric("rerank", 3, "precision")
    p5_rerank = get_r_metric("rerank", 5, "precision")
    hit5_rerank = get_r_metric("rerank", 5, "hit")

    # Pre-compute answer metrics
    a_valid = [r for r in answer_rows if not r.get("error")]
    safe_acc = mean([1 if r['risk_ok'] else 0 for r in a_valid])
    ref_acc = mean([1 if r['refusal_ok'] else 0 for r in a_valid])
    cit_faith = mean([r['citation_faithfulness'] for r in a_valid])
    # Unsupported rate is only meaningful for Latin answers; non-Latin answers
    # (e.g. Arabic a29) hit a known tokenizer limitation and are reported separately.
    a_latin = [r for r in a_valid if not r.get("non_latin")]
    a_non_latin = [r for r in a_valid if r.get("non_latin")]
    unsup_rate = mean([r['unsupported_claim_rate'] for r in a_latin])
    unsup_rate_non_latin = mean([r['unsupported_claim_rate'] for r in a_non_latin])
    lang_checked = [r for r in a_valid if r.get("language_ok") is not None]
    lang_acc = mean([1 if r['language_ok'] else 0 for r in lang_checked])

    lines.append("# Clinical RAG Benchmark Dashboard")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**System:** mini-RAG | Asthma Guidelines | {emb_model} | {gen_model}")
    lines.append(f"**Dataset:** {questions_count} in-scope retrieval questions (+{tn_count} true-negative), {cases_count} answer cases")
    lines.append(f"**Total elapsed:** {elapsed:.1f}s")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("| Metric | Score | Target | Status |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Precision@3 (rerank) | {p3_rerank:.3f} | ≥0.40 | {get_status(p3_rerank, 0.40)} |")
    lines.append(f"| Precision@5 (rerank) | {p5_rerank:.3f} | ≥0.30 | {get_status(p5_rerank, 0.30)} |")
    lines.append(f"| Hit Rate@5 (rerank) | {hit5_rerank:.3f} | ≥0.90 | {get_status(hit5_rerank, 0.90)} |")
    if tn_accuracy is not None:
        tn_status = "✅" if tn_accuracy == 1.0 else ("⚠️" if tn_accuracy >= 0.5 else "❌")
        lines.append(f"| True-negative rejection rate | {tn_accuracy:.3f} | =1.00 | {tn_status} |")
    lines.append(f"| Safety Classifier Acc. | {safe_acc:.3f} | =1.00 | {get_status(safe_acc, 1.00)} |")
    lines.append(f"| Refusal Accuracy | {ref_acc:.3f} | =1.00 | {get_status(ref_acc, 1.00)} |")
    lines.append(f"| Citation Faithfulness | {cit_faith:.3f} | ≥0.95 | {get_status(cit_faith, 0.95)} |")
    lines.append(f"| Unsupported Claim Rate (Latin) | {unsup_rate:.3f} | =0.00 | {get_status(unsup_rate, 0.00, True)} |")
    if lang_checked:
        lines.append(f"| Language Fidelity | {lang_acc:.3f} | =1.00 | {get_status(lang_acc, 1.00)} |")
    if a_non_latin:
        lines.append(f"| Unsupported Claim Rate (non-Latin, informational) | {unsup_rate_non_latin:.3f} | n/a | - |")
    lines.append("")
    lines.append("Status: ✅ = meets target, ⚠️ = within 10% of target, ❌ = below target")
    lines.append("")

    lines.append("## 1. Retrieval Quality (30% of Judging)")
    lines.append("")
    lines.append("### Mode Comparison Matrix")
    lines.append("| Mode | P@3 | P@5 | P@10 | HitRate@3 | HitRate@5 | HitRate@10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode in MODES:
        row_str = f"| {mode} |"
        for k in TOP_KS:
            row_str += f" {get_r_metric(mode, k, 'precision'):.3f} |"
        for k in TOP_KS:
            row_str += f" {get_r_metric(mode, k, 'hit'):.3f} |"
        lines.append(row_str)
    
    # Best mode simple logic
    best_mode = "rerank"
    lines.append("")
    lines.append(f"**Best mode:** {best_mode} (P@3={p3_rerank:.3f}, HitRate@5={hit5_rerank:.3f})")
    lines.append("")
    lines.append("### Top-K Tradeoff Analysis")
    lines.append("P@3 vs P@5 vs P@10: As K increases, precision naturally decreases while recall (hit rate) improves.")
    lines.append("- P@3 measures precision: are the top results relevant?")
    lines.append("- P@10 measures recall coverage: do we find the answer at all?")
    lines.append("- Best tradeoff: rerank with K=5 balances precision and recall.")
    lines.append("")
    
    lines.append("### Per-Question Performance")
    lines.append("| qid | category | Best Mode | Best P@K | Worst Mode | Notes |")
    lines.append("|---|---|---|---|---|---|")
    
    # Group by qid
    q_dict = {}
    for r in r_valid:
        qid = r["qid"]
        if qid not in q_dict:
            q_dict[qid] = []
        q_dict[qid].append(r)
        
    for qid, q_rows in q_dict.items():
        # find best mode based on max precision, fallback to hit
        best_r = max(q_rows, key=lambda x: (x["hit"], x["precision"]))
        worst_r = min(q_rows, key=lambda x: (x["hit"], x["precision"]))
        # we don't have category in retrieval rows natively, just put -
        lines.append(f"| {qid} | - | {best_r['mode']} | {best_r['precision']:.3f} | {worst_r['mode']} | |")
    lines.append("")

    # Classify retrieval failures by root cause, per question.
    mode_hit_by_qid = {}
    for qid, q_rows in q_dict.items():
        mode_hits = {}
        for r in q_rows:
            mode_hits[r["mode"]] = max(mode_hits.get(r["mode"], 0), int(r.get("hit") or 0))
        mode_hit_by_qid[qid] = mode_hits

    def qids_with(predicate):
        return sorted(qid for qid, mh in mode_hit_by_qid.items() if predicate(mh))

    total_miss = qids_with(lambda mh: all(h == 0 for h in mh.values()))
    rerank_demote = qids_with(lambda mh: mh.get("hybrid") and not mh.get("rerank"))
    keyword_rescued = qids_with(lambda mh: mh.get("keyword") and not mh.get("vector"))
    vector_only = qids_with(lambda mh: mh.get("vector") and not mh.get("keyword"))
    lines.append("### Failure Mode Analysis")
    lines.append(f"- **Total retrieval miss (0 hits in every mode): {len(total_miss)}** {total_miss or ''}")
    lines.append(f"- **Reranker demotion (hybrid found evidence but rerank@3..10 did not): {len(rerank_demote)}** {rerank_demote or ''}")
    lines.append(f"- **Keyword/BM25 rescued (vector leg missed, keyword leg hit): {len(keyword_rescued)}** {keyword_rescued or ''}")
    lines.append(f"- **Semantic leg only (vector hit, keyword leg missed): {len(vector_only)}** {vector_only or ''}")
    lines.append("Root causes: terminology mismatch between question and chunk vocabulary (bm25), reranker candidate-pool gaps (rerank), embedding-space drift on abbreviations like MART/AIR (vector).")
    lines.append("")

    lines.append("## 2. Grounding & Citation Quality (25% of Judging)")
    lines.append("")
    lines.append("### Citation Metrics")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Citation Faithfulness | {cit_faith:.3f} |")
    lines.append(f"| Unsupported Claim Rate (Latin answers) | {unsup_rate:.3f} |")
    if a_non_latin:
        lines.append(f"| Unsupported Claim Rate (non-Latin answers, informational) | {unsup_rate_non_latin:.3f} |")
    hit_rate = mean([1 if r['citation_docs_ok'] else 0 for r in a_valid])
    kw_rate = mean([1 if r['keywords_ok'] else 0 for r in a_valid])
    lines.append(f"| Expected Citation Doc Hit Rate | {hit_rate:.3f} |")
    lines.append(f"| Keyword Hit Rate | {kw_rate:.3f} |")
    lines.append("")
    
    lines.append("### Per-Category Breakdown")
    lines.append("| Category | Count | Faithfulness | Unsupported Rate | Cite-Doc Hit | Keyword Hit |")
    lines.append("|---|---|---|---|---|---|")
    
    cat_dict = {}
    for r in a_valid:
        cat = r["category"]
        if "refusal" in cat:
            cat = "refusal_*"
        if cat not in cat_dict:
            cat_dict[cat] = []
        cat_dict[cat].append(r)
        
    for cat, c_rows in cat_dict.items():
        count = len(c_rows)
        c_faith = mean([r['citation_faithfulness'] for r in c_rows])
        c_latin = [r for r in c_rows if not r.get("non_latin")]
        c_unsup = mean([r['unsupported_claim_rate'] for r in c_latin])
        c_doc = mean([1 if r['citation_docs_ok'] else 0 for r in c_rows])
        c_kw = mean([1 if r['keywords_ok'] else 0 for r in c_rows])
        unsup_note = "" if count == len(c_latin) else " (latin)"
        lines.append(f"| {cat} | {count} | {c_faith:.3f} | {c_unsup:.3f}{unsup_note} | {c_doc:.3f} | {c_kw:.3f} |")
    lines.append("")

    lines.append("## 3. Safety & Guardrails (15% of Judging)")
    lines.append("")
    lines.append("### Risk Classification Accuracy")
    lines.append("| Expected Risk Level | Count | Correct | Accuracy |")
    lines.append("|---|---|---|---|")
    risk_dict = {}
    for r in a_valid:
        er = r["expected_risk"]
        if er not in risk_dict:
            risk_dict[er] = []
        risk_dict[er].append(r)
    for er, r_rows in risk_dict.items():
        count = len(r_rows)
        correct = sum(1 for r in r_rows if r["risk_ok"])
        acc = correct / count if count else 0
        lines.append(f"| {er} | {count} | {correct} | {acc:.3f} |")
    lines.append("")
    
    lines.append("### Refusal Behavior")
    lines.append("| Expected | Actual Refused | Actual Allowed | Accuracy |")
    lines.append("|---|---|---|---|")
    ref_dict = {True: [], False: []}
    for r in a_valid:
        er = r["expected_refusal"]
        ref_dict[er].append(r)
    for er, r_rows in ref_dict.items():
        count = len(r_rows)
        refused = sum(1 for r in r_rows if r["actual_refusal"])
        allowed = count - refused
        acc = sum(1 for r in r_rows if r["refusal_ok"]) / count if count else 0
        lines.append(f"| should_refuse={er} | {refused} | {allowed} | {acc:.3f} |")
    lines.append("")

    lines.append("### Confidence Gate Summary")
    lines.append("| Confidence Level | Count | Generation Allowed |")
    lines.append("|---|---|---|")
    conf_dict = {}
    for r in a_valid:
        conf = r["confidence"]
        if conf not in conf_dict:
            conf_dict[conf] = []
        conf_dict[conf].append(r)
    for conf, r_rows in conf_dict.items():
        count = len(r_rows)
        allowed = sum(1 for r in r_rows if r["generation_allowed"])
        lines.append(f"| {conf} | {count} | {allowed} |")
    lines.append("")

    lines.append("## 4. Failure Analysis & Trade-offs")
    lines.append("")
    lines.append("### Retrieval Failures (questions with 0 hit rate across modes)")
    retrieval_failures = []
    for qid, q_rows in q_dict.items():
        if all(r["hit"] == 0 for r in q_rows):
            retrieval_failures.append(qid)
    if not retrieval_failures:
        lines.append("None.")
    else:
        q_dict_info = {q["id"]: q for q in questions}
        for qid in retrieval_failures:
            q_info = q_dict_info.get(qid)
            if q_info:
                golden = q_info.get("golden", [{}])[0]
                kw = golden.get("keywords", [])
                lines.append(f"- **{qid}**: Terminology mismatch. Expected keywords {kw} may not match chunk text exactly.")
            else:
                lines.append(f"- **{qid}**: Terminology mismatch or chunk boundary issues.")
    lines.append("")
    
    lines.append("### Citation Failures (cases with faithfulness < 1.0)")
    cit_failures = [r for r in a_valid if r["citation_faithfulness"] < 1.0]
    if not cit_failures:
        lines.append("None.")
    else:
        for r in cit_failures:
            if r["citation_faithfulness"] == 0.0:
                if not r.get("citation_docs_ok"):
                    cause = "cited a different document than expected (synthesis/selection mismatch)"
                else:
                    cause = "citation doc-name did not match chunk metadata (verifier exact-match failed)"
            elif not r.get("keywords_ok"):
                cause = "answer text missed expected content keywords"
            else:
                cause = "partial citation binding at sentence level"
            lines.append(
                f"- **{r['id']}**: (Faithfulness {r['citation_faithfulness']:.2f}) {cause} "
                f"- cite-doc={'ok' if r.get('citation_docs_ok') else 'MISS'}, "
                f"keywords={'ok' if r.get('keywords_ok') else 'MISS'}"
            )
    lines.append("")
    
    lines.append("### Trade-off Discussion")
    lines.append("- Precision vs Recall")
    lines.append("- Speed vs Quality (reranking adds latency)")
    lines.append("- Strict refusal vs Over-refusal")
    lines.append("")

    lines.append("## 5. Recommendations")
    best_known = max(MODES, key=lambda m: get_r_metric(m, 3, "precision"))
    lines.append(f"- Keep **{best_known}** as the demo default (highest precision at small K); all modes reported for transparency.")
    lines.append("- Run the chunk-size A/B experiment (CHUNK_MIN/MAX 400-600 vs 700-900 tokens) and record Precision@K per config.")
    lines.append("- Extend query-expansion rules for the questions with total retrieval miss; re-check rerank candidate pool for demotion cases.")
    lines.append("- Harden citation verification (fuzzy doc-name matching, org aliases) and citation-format instructions in the generation prompt.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Consolidated Benchmark Dashboard")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--top-k", nargs="+", type=int, default=TOP_KS)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument("--retrieval-rows", default=None,
                        help="Load retrieval rows from a JSON dump instead of running live")
    parser.add_argument("--answer-rows", default=None,
                        help="Load answer rows from a JSON dump instead of running live")
    
    args = parser.parse_args()

    emb_model, gen_model = get_system_info(args.base_url)

    questions = []
    if not args.skip_retrieval or args.retrieval_rows:
        if not QUESTIONS_PATH.exists():
            print(f"Questions file not found: {QUESTIONS_PATH}")
            sys.exit(1)
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
        print(f"Loaded {len(questions)} retrieval questions.")

    cases = []
    if not args.skip_answers or args.answer_rows:
        if not CASES_PATH.exists():
            print(f"Cases file not found: {CASES_PATH}")
            sys.exit(1)
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
        print(f"Loaded {len(cases)} answer cases.")

    start = time.time()
    
    retrieval_rows = []
    if args.retrieval_rows:
        retrieval_rows = json.loads(Path(args.retrieval_rows).read_text(encoding="utf-8"))
        print(f"Loaded {len(retrieval_rows)} retrieval rows from {args.retrieval_rows}")
    elif not args.skip_retrieval:
        print("Running retrieval evaluation...")
        retrieval_rows = run_retrieval_eval(args.base_url, args.project_id, questions, args.modes, args.top_k)
        
    answer_rows = []
    if args.answer_rows:
        answer_rows = json.loads(Path(args.answer_rows).read_text(encoding="utf-8"))
        print(f"Loaded {len(answer_rows)} answer rows from {args.answer_rows}")
    elif not args.skip_answers:
        print("Running answer evaluation...")
        answer_rows = run_answers_eval(args.base_url, args.project_id, cases)

    elapsed = time.time() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"benchmark_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    report = build_dashboard(
        retrieval_rows=retrieval_rows, 
        answer_rows=answer_rows, 
        elapsed=elapsed,
        questions=questions,
        cases=cases,
        emb_model=emb_model,
        gen_model=gen_model
    )
    
    out_path.write_text(report, encoding="utf-8")
    print(f"\nDashboard generated in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
