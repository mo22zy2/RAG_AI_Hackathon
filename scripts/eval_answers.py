"""
Answer-quality evaluation harness.

Runs eval/answer_cases_v2.json against a running API server and reports:
    - safety classifier accuracy
    - refusal accuracy
    - citation faithfulness
    - unsupported claim rate (Latin answers scored; non-Latin reported separately)
    - language fidelity for cases with an expected_language (e.g. Arabic a29)
    - confidence-gate summary

Also dumps full answer + citations + source pages for every case to
eval/answer_diagnostics_<timestamp>.json so failing cases can be diagnosed.

Usage (run from src/):
    python ../scripts/eval_answers.py --project-id 1
    python ../scripts/eval_answers.py --project-id 1 --base-url http://localhost:8000
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = REPO_ROOT / "eval" / "answer_cases_v2.json"
OUTPUT_DIR = REPO_ROOT / "eval"

# Arabic script range (Arabic block). Used for language fidelity / a29 checks.
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
# Latin-script answers are those with no Arabic script. English doc names and
# page numbers embedded in an Arabic answer don't disqualify it.
NON_LATIN_LANGUAGES = ("ar",)


def detect_language(text):
    """Return 'ar' if the text contains Arabic script, else 'en' (or 'unknown')."""
    if not (text or "").strip():
        return "unknown"
    return "ar" if _ARABIC_RE.search(text or "") else "en"


def call_answer(base_url, project_id, payload):
    url = f"{base_url.rstrip('/')}/api/v1/nlp/index/answer/{project_id}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def contains_any(text, values):
    if not values:
        return True
    lowered = (text or "").lower()
    return any(value.lower() in lowered for value in values)


def citation_docs_hit(citations, expected_docs):
    if not expected_docs:
        return True
    cited = " ".join(
        f"{item.get('document', '')} {item.get('citation', '')}"
        for item in citations or []
    ).lower()
    return any(doc.lower() in cited for doc in expected_docs)


def run_eval(base_url, project_id, cases):
    rows = []
    for case in cases:
        payload = {
            "text": case["question"],
            "limit": 8,
            "retrieval_mode": "hybrid",
            "rerank": True,
            "expand_query": True,
            "verify_claims": True,
        }
        try:
            body = call_answer(base_url, project_id, payload)
        except Exception as exc:
            rows.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "error": str(exc),
            })
            continue

        answer = body.get("answer") or ""
        risk = body.get("risk_assessment") or {}
        confidence = body.get("confidence") or {}
        quality = body.get("quality") or {}
        citations = quality.get("citations") or body.get("citations") or []
        expected_refusal = bool(case.get("should_refuse"))
        actual_refusal = risk.get("risk_level") == "refuse_redirect" or not confidence.get("generation_allowed", True)

        language = detect_language(answer)
        expected_language = case.get("expected_language")
        language_ok = None
        if expected_language:
            language_ok = language == expected_language

        rows.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "error": None,
            "expected_risk": case.get("expected_risk_level"),
            "actual_risk": risk.get("risk_level"),
            "risk_ok": risk.get("risk_level") == case.get("expected_risk_level"),
            "expected_refusal": expected_refusal,
            "actual_refusal": actual_refusal,
            "refusal_ok": expected_refusal == actual_refusal,
            "citation_faithfulness": quality.get("citation_faithfulness", 0.0),
            "unsupported_claim_rate": quality.get("unsupported_claim_rate", 0.0),
            "unsupported_count": len(quality.get("unsupported_claims") or []),
            "confidence": confidence.get("confidence_level"),
            "generation_allowed": confidence.get("generation_allowed"),
            "citation_docs_ok": True if actual_refusal else citation_docs_hit(citations, case.get("expected_citation_docs")),
            "keywords_ok": True if actual_refusal else contains_any(answer, case.get("expected_keywords")),
            "language": language,
            "expected_language": expected_language,
            "language_ok": language_ok,
            "non_latin": language in NON_LATIN_LANGUAGES,
            "answer_preview": answer[:140].replace("\n", " "),
            # Raw data for the diagnostics dump.
            "answer": answer,
            "citations": citations,
            "unsupported_claims": quality.get("unsupported_claims") or [],
            "sources": body.get("sources") or [],
            "evidence_panel": body.get("evidence_panel"),
            "expanded_query": body.get("expanded_query"),
            "risk_assessment": risk,
            "confidence_detail": confidence,
        })
    return rows


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def build_markdown(cases, rows, elapsed):
    valid = [row for row in rows if not row.get("error")]
    latin = [row for row in valid if not row.get("non_latin")]
    non_latin = [row for row in valid if row.get("non_latin")]
    lang_checked = [row for row in valid if row.get("language_ok") is not None]
    lines = [
        "# Answer Quality Evaluation Results",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Cases:** {len(cases)} | **Successful calls:** {len(valid)} | **Elapsed:** {elapsed:.1f}s",
        "",
        "## Summary",
        "| metric | value |",
        "|---|---:|",
        f"| safety classifier accuracy | {mean([1 if r['risk_ok'] else 0 for r in valid]):.3f} |",
        f"| refusal accuracy | {mean([1 if r['refusal_ok'] else 0 for r in valid]):.3f} |",
        f"| citation faithfulness | {mean([r['citation_faithfulness'] for r in valid]):.3f} |",
        f"| unsupported claim rate (Latin answers) | {mean([r['unsupported_claim_rate'] for r in latin]):.3f} |",
        f"| unsupported claim rate (non-Latin answers, informational) | {mean([r['unsupported_claim_rate'] for r in non_latin]):.3f} |",
        f"| expected citation doc hit rate | {mean([1 if r['citation_docs_ok'] else 0 for r in valid]):.3f} |",
        f"| keyword hit rate | {mean([1 if r['keywords_ok'] else 0 for r in valid]):.3f} |",
        f"| language fidelity (expected_language cases) | {mean([1 if r['language_ok'] else 0 for r in lang_checked]):.3f} |",
        "",
        "## Per-case",
        "| id | category | risk | refusal | lang | faithfulness | unsupported | cite-doc | keywords | error |",
        "|---|---|---|---|---|---:|---:|---|---|---|",
    ]

    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['id']} | {row['category']} | - | - | - | - | - | - | - | {row['error']} |")
            continue
        risk = "yes" if row["risk_ok"] else f"no ({row['actual_risk']})"
        refusal = "yes" if row["refusal_ok"] else f"no ({row['actual_refusal']})"
        lang = f"{row['language']}"
        if row["language_ok"] is not None:
            lang += "/yes" if row["language_ok"] else "/NO"
        else:
            lang += "/-"
        cite_doc = "yes" if row["citation_docs_ok"] else "no"
        keywords = "yes" if row["keywords_ok"] else "no"
        lines.append(
            f"| {row['id']} | {row['category']} | {risk} | {refusal} | {lang} | "
            f"{row['citation_faithfulness']:.3f} | {row['unsupported_claim_rate']:.3f} | "
            f"{cite_doc} | {keywords} |  |"
        )

    lines.extend(["", "## Failure Notes"])
    failures = [
        row for row in valid
        if not row["risk_ok"] or not row["refusal_ok"] or not row["citation_docs_ok"]
        or (row["unsupported_claim_rate"] > 0 and not row["non_latin"])
        or (row["language_ok"] is False)
    ]
    if not failures:
        lines.append("- No failures detected by deterministic checks.")
    else:
        for row in failures:
            extra = []
            if row.get("non_latin"):
                extra.append("non-Latin answer (unsupported rate informational)")
            if row.get("language_ok") is False:
                extra.append(f"language mismatch (expected {row['expected_language']})")
            note = f", {'; '.join(extra)}" if extra else ""
            lines.append(
                f"- **{row['id']}**: risk={row['actual_risk']}, "
                f"confidence={row['confidence']}, unsupported={row['unsupported_count']}, "
                f"preview={row['answer_preview']}{note}"
            )

    lines.append("")
    return "\n".join(lines)


def build_diagnostics(rows):
    """Compact full-answer details for failing (or all) cases to a JSON list."""
    diagnostics = []
    for row in rows:
        if row.get("error"):
            diagnostics.append({"id": row["id"], "question": row["question"], "error": row["error"]})
            continue
        diagnostics.append({
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "expected_risk_level": row["expected_risk"],
            "actual_risk_level": row["actual_risk"],
            "expected_refusal": row["expected_refusal"],
            "actual_refusal": row["actual_refusal"],
            "risk_ok": row["risk_ok"],
            "refusal_ok": row["refusal_ok"],
            "confidence": row["confidence_detail"],
            "risk_assessment": row["risk_assessment"],
            "language": row["language"],
            "language_ok": row["language_ok"],
            "citation_faithfulness": row["citation_faithfulness"],
            "unsupported_claim_rate": row["unsupported_claim_rate"],
            "unsupported_claims": row["unsupported_claims"],
            "citations": row["citations"],
            "sources": [
                {
                    "document_name": s.get("document_name") or s.get("file_name"),
                    "page_number": s.get("page_number"),
                    "section_title": s.get("section_title"),
                    "score": s.get("score"),
                    "text": s.get("text"),
                }
                for s in row.get("sources") or []
            ],
            "evidence_panel": row.get("evidence_panel"),
            "expanded_query": row.get("expanded_query"),
            "answer": row["answer"],
        })
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Answer-quality evaluation")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    if not CASES_PATH.exists():
        print(f"Cases file not found: {CASES_PATH}")
        sys.exit(1)

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    start = time.time()
    rows = run_eval(args.base_url, args.project_id, cases)
    elapsed = time.time() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f"answer_results_{ts}.md"
    out_path.write_text(build_markdown(cases, rows, elapsed), encoding="utf-8")
    diag_path = OUTPUT_DIR / f"answer_diagnostics_{ts}.json"
    diag_path.write_text(
        json.dumps(build_diagnostics(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows_path = OUTPUT_DIR / f"answer_rows_{ts}.json"
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done in {elapsed:.1f}s -> {out_path}")
    print(f"Diagnostics -> {diag_path}")
    print(f"Rows -> {rows_path}")

    errors = [row for row in rows if row.get("error")]
    if errors:
        print(f"WARNING: {len(errors)} failed calls. First error: {errors[0]['error'][:160]}")


if __name__ == "__main__":
    main()
