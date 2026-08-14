"""
Audit golden labels: verify every golden entry in clinical_questions_v2.json
exists in the live pgvector collection.

Checks:
  1. doc_hint matches a document_name in the collection
  2. page_number within [page_from - PAGE_WINDOW, page_to + PAGE_WINDOW]
  3. at least one golden keyword present in chunk text

Reports PASS/FAIL per entry, and lists chunks that match each golden rule.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "clinical_questions_v2.json"
DB = "clinical_rag"
COLLECTION = "collection_1024_1"
PAGE_WINDOW = 2


def query_db(sql: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "pgvector", "psql", "-U", "postgres", "-d", DB, "-t", "-c", sql],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  DB ERROR: {result.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def check_golden(entry: dict) -> dict:
    """Check one golden entry against the DB. Returns {pass, matches, details}."""
    doc_hint = entry.get("doc_hint", "").lower()
    page_from = entry.get("page_from", 0)
    page_to = entry.get("page_to", 0)
    keywords = [k for k in entry.get("keywords", []) if k]

    # Find chunks where document_name contains doc_hint
    doc_rows = query_db(
        f"SELECT chunk_id, metadata->>'document_name' AS doc, "
        f"metadata->>'page_number' AS page, length(text) AS tlen "
        f"FROM {COLLECTION} "
        f"WHERE lower(metadata->>'document_name') LIKE '%{doc_hint}%' "
        f"ORDER BY CAST(metadata->>'page_number' AS INT)"
    )

    chunks = []
    for line in doc_rows.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 4:
            continue
        chunk_id, doc, page, tlen = parts[0], parts[1], parts[2], parts[3]
        try:
            page_int = int(page)
        except ValueError:
            continue
        # Check page window
        if page_int < (page_from - PAGE_WINDOW) or page_int > (page_to + PAGE_WINDOW):
            continue
        # Check keywords via OR clause
        kw_clauses = " OR ".join(
            f"text ILIKE '%{kw.replace(chr(39), '')}%' " for kw in keywords
        )
        kw_check = query_db(
            f"SELECT COUNT(*) FROM {COLLECTION} "
            f"WHERE chunk_id = {chunk_id} AND ({kw_clauses})"
        )
        count = 0
        try:
            count = int(kw_check.strip())
        except ValueError:
            pass
        chunks.append({
            "chunk_id": chunk_id, "doc": doc, "page": page_int,
            "len": tlen, "keyword_hit": count > 0,
        })

    hits = [c for c in chunks if c["keyword_hit"]]
    return {
        "page_matches": len(chunks),
        "keyword_hits": len(hits),
        "pass": len(hits) > 0,
        "matching_chunks": hits,
        "all_page_matches": chunks,
    }


def main():
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]

    print(f"Auditing {len(questions)} questions against {COLLECTION}...")
    print("=" * 72)

    failures = []
    for q in questions:
        qid = q["id"]
        golden_entries = q.get("golden", [])
        if not golden_entries:
            # true-negative / expect_hit=false → no check needed
            if q.get("expect_hit") is False:
                print(f"{qid}: TRUE-NEGATIVE (no golden, expect_hit=false) — SKIP")
                continue
            print(f"{qid}: NO GOLDEN ENTRIES — CHECK MANUALLY")
            failures.append((qid, "no golden entries"))
            continue

        entry_results = []
        all_pass = True
        for i, entry in enumerate(golden_entries):
            result = check_golden(entry)
            status = "PASS" if result["pass"] else "FAIL"
            if not result["pass"]:
                all_pass = False
            entry_results.append((entry, result))

        overall = "PASS" if all_pass else "FAIL"
        if not all_pass:
            failures.append((qid, entry_results))

        # Print summary
        print(f"\n{qid} ({q['category']}): {overall}")
        for entry, result in entry_results:
            doc_hint = entry["doc_hint"]
            pages = f"[{entry['page_from']}-{entry['page_to']}]"
            hits = result["keyword_hits"]
            page_match = result["page_matches"]
            if result["pass"]:
                print(f"  {doc_hint} p.{pages}: {hits} keyword hit(s), {page_match} chunks in window OK")
            else:
                print(f"  {doc_hint} p.{pages}: 0 keyword hits, {page_match} chunks in window FAIL")
                # Show where the actual content lives (all chunks from this doc)
                all_chunks = result["all_page_matches"]
                if all_chunks:
                    print(f"    Chunks found: {[(c['chunk_id'], c['page']) for c in all_chunks[:5]]}")
                # Search broader: what pages have the keywords?
                for kw in entry.get("keywords", []):
                    broad = query_db(
                        f"SELECT chunk_id, metadata->>'page_number' AS page "
                        f"FROM {COLLECTION} "
                        f"WHERE lower(metadata->>'document_name') LIKE '%{doc_hint.lower()}%' "
                        f"AND text ILIKE '%{kw}%' "
                        f"ORDER BY CAST(metadata->>'page_number' AS INT) LIMIT 3"
                    )
                    if broad.strip():
                        print(f"    Keyword '{kw}' found on pages: {broad.strip()[:200]}")

    print("\n" + "=" * 72)
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for qid, detail in failures:
            print(f"  {qid}")
    else:
        print("\nAll labels pass!")

    # Also show the reverse: which chunks exist for docs that were never matched
    print("\n\n=== DOCUMENT COVERAGE ===")
    doc_counts = query_db(
        f"SELECT metadata->>'document_name' AS doc, COUNT(*), MIN(CAST(metadata->>'page_number' AS INT)), MAX(CAST(metadata->>'page_number' AS INT)) "
        f"FROM {COLLECTION} GROUP BY 1 ORDER BY 1"
    )
    print(doc_counts)


if __name__ == "__main__":
    main()
