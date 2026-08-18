"""
Ingest official guideline PDFs into Dataset/source_pdfs/ from Dataset/manifest.json.

Only sources listed in the manifest are allowed. For each entry this script:

  1. Skips entries that already have a valid local file (by `file_name`).
  2. Downloads `source_url` into `Dataset/source_pdfs/<file_name>` (with retries).
  3. Validates the PDF magic bytes.
  4. Fills in `local_path`, `size_bytes`, `sha256`, `page_count`, `downloaded_at`.

Usage (run from the repo root):

    python scripts/ingest_guidelines.py            # download whatever is missing
    python scripts/ingest_guidelines.py --status   # show manifest download state
    python scripts/ingest_guidelines.py --reset    # re-download all entries

Requires PyMuPDF (already a project dependency) for the page count; it is used
only if importable. No other third-party packages are needed.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "Dataset" / "manifest.json"
DEFAULT_SOURCE_DIR = REPO_ROOT / "Dataset" / "source_pdfs"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BreathXRAG-Hackathon/1.0"
DOWNLOAD_TIMEOUT = 90
MAX_RETRIES = 2
PDF_MAGIC = b"%PDF-"


def load_manifest():
    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def is_valid_pdf(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(PDF_MAGIC)) == PDF_MAGIC


def get_page_count(path: Path) -> int:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        pass

    # Dependency-free fallback: count page objects in the raw PDF stream.
    # Loose match on /Type /Page (space-insensitive); fails on fully
    # compressed object streams (page_count stays 0 until PyMuPDF works).
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return max(len(re.findall(rb"/Type\s*/Page[\s>]", data)) - 1, 0)
    except Exception:
        return 0


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                with open(destination, "wb") as out:
                    while True:
                        block = response.read(1 << 16)
                        if not block:
                            break
                        out.write(block)
            return True
        except (HTTPError, URLError) as exc:
            last_error = exc
            print(f"    attempt {attempt} failed: {exc}")
    raise RuntimeError(f"download failed for {url}: {last_error}")


def fill_metadata(entry: dict, destination: Path):
    entry["local_path"] = str(destination.relative_to(REPO_ROOT)).replace("\\", "/")
    entry["size_bytes"] = destination.stat().st_size
    entry["sha256"] = sha256_of(destination)
    entry["page_count"] = get_page_count(destination)


def ingest_entry(entry: dict, source_dir: Path, reset: bool = False):
    file_name = entry.get("file_name")
    url = entry.get("source_url", "").strip()
    entry_id = entry.get("id", file_name or "?")

    if not file_name:
        print(f"[skip] {entry_id}: no file_name in manifest")
        return False

    destination = source_dir / file_name
    if reset and destination.exists():
        destination.unlink()

    if destination.exists() and is_valid_pdf(destination):
        fill_metadata(entry, destination)
        print(
            f"[ok]   {entry_id}: {destination.name} "
            f"({entry['page_count']} pages, sha256={entry['sha256'][:12]}...)"
        )
        return True

    if not url:
        print(f"[wait] {entry_id}: file missing and source_url is empty, nothing to do")
        return False

    print(f"[get]  {entry_id}: {url}")
    download_file(url, destination)

    if not is_valid_pdf(destination):
        print(f"[fail] {entry_id}: downloaded file is not a valid PDF")
        destination.unlink()
        return False

    fill_metadata(entry, destination)
    print(
        f"[done] {entry_id}: {entry['page_count']} pages, "
        f"{entry['size_bytes']} bytes, sha256={entry['sha256'][:12]}..."
    )
    return True


def print_status(manifest, source_dir: Path):
    print(f"Topic: {manifest.get('topic')}")
    print(f"{'ID':<34} {'status':<12} {'pages':>6}  file")
    print("-" * 90)
    for entry in manifest["entries"]:
        file_name = entry.get("file_name", "")
        path = source_dir / file_name
        if path.exists() and is_valid_pdf(path):
            status = "downloaded"
        elif entry.get("source_url", "").strip():
            status = "missing"
        else:
            status = "no url"
        print(
            f"{entry.get('id', '?'):<34} {status:<12} "
            f"{str(entry.get('page_count') or 0):>6}  {file_name}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="only print download state")
    parser.add_argument("--reset", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    manifest = load_manifest()
    source_dir = REPO_ROOT / manifest.get("source_dir", DEFAULT_SOURCE_DIR.name)
    source_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status(manifest, source_dir)
        return

    changed = False
    for entry in manifest["entries"]:
        if ingest_entry(entry, source_dir, reset=args.reset):
            changed = True

    if changed:
        save_manifest(manifest)
        print("\nManifest updated.")
    else:
        print("\nNothing new downloaded.")


if __name__ == "__main__":
    main()
