"""
Chunk diagnostics report.

Inspects stored chunks for a project and reports chunk sizes, metadata health,
page coverage, and source provenance.

Usage (run from src/):
    python ../scripts/chunk_diagnostics.py --project-id 1
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from helpers.config import get_settings
from models.db_schemas import Asset, DataChunk
OUTPUT_DIR = REPO_ROOT / "eval"
REQUIRED_METADATA = ["document_name", "source_url", "page_number", "section_title", "chunk_id"]


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def estimate_tokens(text, chars_per_token=4.0):
    return max(int(len(text or "") / chars_per_token), 1)


def percentile(values, ratio):
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, int((len(values) - 1) * ratio))
    return values[index]


async def load_project_data(project_id):
    settings = get_settings()
    db_url = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(db_url)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with db_client() as session:
        chunks_result = await session.execute(
            select(DataChunk).where(DataChunk.chunk_project_id == project_id).order_by(DataChunk.chunk_asset_id, DataChunk.chunk_order)
        )
        assets_result = await session.execute(
            select(Asset).where(Asset.asset_project_id == project_id)
        )
        chunks = chunks_result.scalars().all()
        assets = assets_result.scalars().all()

    await engine.dispose()
    return assets, chunks


def build_report(project_id, assets, chunks):
    tokens = [estimate_tokens(chunk.chunk_text) for chunk in chunks]
    by_asset = defaultdict(list)
    missing_counts = defaultdict(int)
    pages_by_asset = defaultdict(set)
    source_urls = set()
    document_names = set()
    empty_chunks = 0

    for chunk in chunks:
        meta = parse_metadata(chunk.chunk_metadata)
        meta.setdefault("chunk_id", chunk.chunk_id)
        by_asset[chunk.chunk_asset_id].append(chunk)
        if not (chunk.chunk_text or "").strip():
            empty_chunks += 1
        for key in REQUIRED_METADATA:
            if meta.get(key) in (None, "", 0):
                missing_counts[key] += 1
        if meta.get("page_number") not in (None, "", 0):
            pages_by_asset[chunk.chunk_asset_id].add(meta.get("page_number"))
        if meta.get("source_url"):
            source_urls.add(meta["source_url"])
        if meta.get("document_name"):
            document_names.add(meta["document_name"])

    lines = [
        "# Chunk Diagnostics",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Project ID:** {project_id}",
        "",
        "## Summary",
        "| metric | value |",
        "|---|---:|",
        f"| assets | {len(assets)} |",
        f"| chunks | {len(chunks)} |",
        f"| empty chunks | {empty_chunks} |",
        f"| token min | {min(tokens) if tokens else 0} |",
        f"| token median | {percentile(tokens, 0.5)} |",
        f"| token p90 | {percentile(tokens, 0.9)} |",
        f"| token max | {max(tokens) if tokens else 0} |",
        f"| unique documents | {len(document_names)} |",
        f"| source URLs | {len(source_urls)} |",
        "",
        "## Metadata Completeness",
        "| field | missing chunks |",
        "|---|---:|",
    ]

    for key in REQUIRED_METADATA:
        lines.append(f"| {key} | {missing_counts[key]} |")

    lines.extend([
        "",
        "## Per Asset",
        "| asset_id | file_name | chunks | page coverage | document_name | source_url |",
        "|---|---|---:|---:|---|---|",
    ])

    assets_by_id = {asset.asset_id: asset for asset in assets}
    for asset_id, asset_chunks in sorted(by_asset.items()):
        asset = assets_by_id.get(asset_id)
        sample_meta = parse_metadata(asset_chunks[0].chunk_metadata) if asset_chunks else {}
        file_name = asset.asset_name if asset else sample_meta.get("file_name", "")
        document_name = sample_meta.get("document_name", "")
        source_url = sample_meta.get("source_url", "")
        lines.append(
            f"| {asset_id} | {file_name} | {len(asset_chunks)} | "
            f"{len(pages_by_asset[asset_id])} | {document_name} | {source_url} |"
        )

    lines.append("")
    return "\n".join(lines)


async def main_async(project_id):
    assets, chunks = await load_project_data(project_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"chunk_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text(build_report(project_id, assets, chunks), encoding="utf-8")
    print(f"Done -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Chunk diagnostics report")
    parser.add_argument("--project-id", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args.project_id))


if __name__ == "__main__":
    main()
