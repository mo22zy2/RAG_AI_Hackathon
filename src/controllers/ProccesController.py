from typing import List, Optional
from dataclasses import dataclass, field
from collections import Counter
import os
import re
import math

from .BaseController import BaseController
from .ProjectController import ProjectController
from langchain_community.document_loaders import TextLoader  # type: ignore
from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from models import ProcessingEnum


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class ProcessController(BaseController):

    def __init__(self, project_id: int):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):
        file_ext = self.get_file_extension(file_id)
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")

        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)

        return None

    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()
        return None

    def process_file_content(
        self,
        file_content: list,
        file_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function=len,
        method: str = "section",  # "section" | "recursive" | "simple" | "semantic"
        embeddings=None,  # required when method="semantic"; any object with .embed_documents(List[str]) -> List[List[float]]
        semantic_breakpoint_percentile: float = 95.0,
        semantic_min_chunk_size: int = 0,
    ):
        """
        Chunk loaded documents.

        Every method first runs a cleaning pass (running header/footer lines and
        config-driven artifact patterns are stripped) and a metadata pass that
        normalizes `page_number` (1-indexed) and seeds `section_title`.

        method="section" (default): section-aware splitter — detects section
            headings (numbered, ALL-CAPS, or config patterns), keeps each section
            in context, and splits oversized sections into chunks within the
            configured token budget (CHUNK_MIN/MAX_TOKENS).
        method="recursive": LangChain's RecursiveCharacterTextSplitter —
            sentence/paragraph-aware, real overlap. Good general-purpose default.
        method="simple": the custom line-based fallback splitter.
        method="semantic": groups sentences by embedding similarity so each chunk
            stays topically coherent, instead of cutting at a fixed size. Costs an
            embedding call per document at ingest time. Requires `embeddings`.
        """
        cleaned_content = self._clean_pages(file_content)
        file_content_texts = [rec.page_content for rec in cleaned_content]
        file_content_metadata = [
            self._normalize_metadata(rec, file_id) for rec in cleaned_content
        ]

        if method == "section":
            return self.process_section_splitter(
                file_content_texts,
                file_content_metadata,
                min_tokens=self.app_settings.CHUNK_MIN_TOKENS,
                max_tokens=self.app_settings.CHUNK_MAX_TOKENS,
                chunk_overlap=chunk_overlap,
            )

        if method == "semantic":
            if embeddings is None:
                raise ValueError(
                    "process_file_content(method='semantic') requires an `embeddings` "
                    "object exposing .embed_documents(list[str]) -> list[list[float]] "
                    "(e.g. a LangChain Embeddings instance)."
                )
            return self.process_semantic_splitter(
                file_content_texts,
                file_content_metadata,
                embeddings=embeddings,
                breakpoint_percentile=semantic_breakpoint_percentile,
                min_chunk_size=semantic_min_chunk_size or chunk_size // 4,
                max_chunk_size=chunk_size * 4,  # safety cap so no chunk runs away
            )

        if method == "simple":
            return self.process_simpler_splitter(
                file_content_texts,
                file_content_metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            # sensible separator priority: paragraph -> line -> sentence -> word -> char
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return text_splitter.create_documents(
            file_content_texts, metadatas=file_content_metadata
        )

    # ------------------------------------------------------------------ #
    # cleaning + metadata normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_metadata(record, file_id: str) -> dict:
        """Normalize per-page loader metadata; adds `page_number` (1-indexed)
        and seeds `section_title` so downstream chunkers can enrich it."""
        meta = dict(record.metadata or {})
        meta["file_id"] = file_id
        page = meta.get("page")
        try:
            meta["page_number"] = int(page) + 1 if page is not None else 0
        except (ValueError, TypeError):
            meta["page_number"] = 0
        meta.setdefault("section_title", "")
        return meta

    def _clean_pages(self, content: list) -> list:
        """
        Strip extraction artifacts before chunking:
          1. running headers/footers — lines that repeat on >=
             RUNNING_HEADER_MIN_PAGE_FRACTION of pages (e.g. "GINA 2026", page numbers)
          2. STRIP_PATTERNS — config-driven regexes matched against each line
        """
        pages = [(rec.page_content or "", dict(rec.metadata or {})) for rec in content]
        if not pages:
            return [Document(page_content="", metadata={})]

        fraction = self.app_settings.RUNNING_HEADER_MIN_PAGE_FRACTION
        page_count = max(len(pages), 1)
        running = set()

        first_lines = []
        last_lines = []
        for text, _ in pages:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            first_lines.append(lines[0] if lines else "")
            last_lines.append(lines[-1] if lines else "")

        for column in (first_lines, last_lines):
            for line, count in Counter(column).items():
                if line and len(line) < 120 and count / page_count >= fraction:
                    running.add(line)

        compiled = self._compile_patterns(self.app_settings.STRIP_PATTERNS)

        cleaned = []
        for text, meta in pages:
            kept = []
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                if line in running:
                    continue
                if any(rx.search(raw_line) for rx in compiled):
                    continue
                kept.append(raw_line.rstrip())
            cleaned.append(
                Document(page_content="\n".join(kept).strip(), metadata=meta)
            )
        return cleaned

    @staticmethod
    def _compile_patterns(patterns) -> List[re.Pattern]:
        compiled = []
        for pattern in patterns or []:
            try:
                compiled.append(re.compile(pattern))
            except re.error:
                continue
        return compiled

    def process_simpler_splitter(
        self,
        texts: List[str],
        metadatas: List[dict],
        chunk_size: int,
        chunk_overlap: int = 0,
        splitter_tag: str = "\n",
    ):
        """
        Custom fallback splitter. Processes each source text independently
        (so documents never bleed into each other) and preserves metadata
        and overlap between chunks.
        """
        chunks = []

        for text, metadata in zip(texts, metadatas):
            lines = [
                line.strip()
                for line in text.split(splitter_tag)
                if len(line.strip()) > 1
            ]

            current_chunk = ""
            for line in lines:
                current_chunk += line + splitter_tag

                if len(current_chunk) >= chunk_size:
                    stripped = current_chunk.strip()
                    chunks.append(Document(page_content=stripped, metadata=dict(metadata)))

                    # carry the tail of this chunk into the next one for overlap
                    if chunk_overlap > 0:
                        current_chunk = stripped[-chunk_overlap:] + splitter_tag
                    else:
                        current_chunk = ""

            if len(current_chunk.strip()) > 0:
                chunks.append(
                    Document(page_content=current_chunk.strip(), metadata=dict(metadata))
                )

        return chunks

    # -------------------- section-aware chunking --------------------

    def process_section_splitter(
        self,
        texts: List[str],
        metadatas: List[dict],
        min_tokens: int = 400,
        max_tokens: int = 800,
        chunk_overlap: int = 0,
    ):
        """
        Section-aware, token-budgeted splitter.

        Detects section headings (numbered like "1.1 Diagnosis", ALL-CAPS lines,
        or config SECTION_HEADER_PATTERNS), keeps every chunk tagged with its
        active `section_title`, and splits oversized sections into chunks within
        the [min_tokens, max_tokens] budget. `page_number` is attached per chunk
        (majority page of the chunk's lines).
        """
        heading_patterns = self._compile_patterns(
            self.app_settings.SECTION_HEADER_PATTERNS
        )
        sections = self._extract_sections(texts, metadatas, heading_patterns)
        chunks = self._chunk_sections(sections, max_tokens, chunk_overlap)
        return self._merge_small_chunks(chunks, min_tokens, max_tokens)

    def _extract_sections(self, texts, metadatas, heading_patterns):
        sections = []
        current_title = ""
        current_items = []

        for text, meta in zip(texts, metadatas):
            page = meta.get("page_number") or 0
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                heading = self._detect_heading(line, heading_patterns)
                if heading is not None:
                    if current_items:
                        sections.append((current_items, current_title))
                    current_title = heading
                    current_items = []
                current_items.append((line, page))

        if current_items:
            sections.append((current_items, current_title))

        return sections

    def _chunk_sections(self, sections, max_tokens: int, chunk_overlap: int):
        chunks = []

        for items, section_title in sections:
            current_lines = []
            current_pages = []
            current_tokens = 0

            def flush():
                nonlocal current_lines, current_pages, current_tokens
                if current_lines:
                    text = "\n".join(current_lines).strip()
                    if text:
                        page_number = (
                            Counter(current_pages).most_common(1)[0][0]
                            if current_pages
                            else 0
                        )
                        chunks.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "section_title": section_title or "",
                                    "page_number": page_number,
                                },
                            )
                        )
                if chunk_overlap > 0 and current_lines:
                    tail = " ".join(current_lines)[-chunk_overlap:]
                    current_lines = [tail] if tail.strip() else []
                    current_pages = [current_pages[-1]] if current_pages else []
                    current_tokens = self._estimate_tokens(tail)
                else:
                    current_lines = []
                    current_pages = []
                    current_tokens = 0

            for line, page in items:
                line_tokens = self._estimate_tokens(line)
                if current_lines and current_tokens + line_tokens > max_tokens:
                    flush()
                current_lines.append(line)
                current_pages.append(page)
                current_tokens += line_tokens

            flush()

        return chunks

    def _merge_small_chunks(self, chunks: list, min_tokens: int, max_tokens: int):
        """
        Post-pass for the section splitter: table-heavy documents (e.g. quick
        reference guides) often leave single-word fragments as standalone
        chunks. Any chunk below `min_tokens` is merged with the following
        chunks until it reaches `min_tokens` (or would exceed `max_tokens`).
        Page/section metadata is kept from the portions contributing the most
        characters.
        """
        merged = []
        i = 0
        n = len(chunks)

        while i < n:
            chunk = chunks[i]
            if self._estimate_tokens(chunk.page_content) >= min_tokens:
                merged.append(chunk)
                i += 1
                continue

            text = chunk.page_content
            pages_weight = {}
            sections_weight = {}

            def absorb(segment):
                nonlocal text
                text = text + "\n" + segment.page_content
                page = segment.metadata.get("page_number") or 0
                pages_weight[page] = pages_weight.get(page, 0) + len(segment.page_content)
                section = segment.metadata.get("section_title") or ""
                if section:
                    sections_weight[section] = (
                        sections_weight.get(section, 0) + len(segment.page_content)
                    )

            absorb(chunk)
            j = i + 1
            while j < n and self._estimate_tokens(text) < min_tokens:
                next_chunk = chunks[j]
                candidate_tokens = self._estimate_tokens(
                    text + "\n" + next_chunk.page_content
                )
                if candidate_tokens > max_tokens:
                    break
                absorb(next_chunk)
                j += 1

            page_number = max(pages_weight, key=pages_weight.get) if pages_weight else 0
            section_title = (
                max(sections_weight, key=sections_weight.get) if sections_weight else ""
            )
            merged.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": page_number,
                        "section_title": section_title,
                    },
                )
            )
            i = j

        return merged

    @staticmethod
    def _detect_heading(line: str, compiled_patterns) -> Optional[str]:
        """Return a cleaned section title if `line` looks like a heading, else None."""
        s = line.strip()
        if not s or len(s) < 4 or len(s) > 120:
            return None
        if s.isdigit() or re.fullmatch(r"[\d\.\s\-–—]+", s):
            return None

        # numbered section heading: "1.1 Diagnosis", "5 Recommendations",
        # "3 Management: overview" — exclude lines that read as a sentence.
        numbered = re.match(r"^\d+(\.\d+)*[\.\)]?\s+\S", s)
        if numbered and not s.rstrip().endswith((".", "!", "?")):
            cleaned = re.sub(r"^\d+(\.\d+)*[\.\)]?\s+", "", s).strip()
            return cleaned or s

        # ALL-CAPS line, e.g. "POCKET GUIDE FOR ASTHMA MANAGEMENT".
        # Require a multi-word line OR a longer single-word one so short table
        # labels like "ASSESS" / "STEP 1" are not mistaken for headings.
        if (
            s.isupper()
            and len(s) <= 80
            and sum(1 for c in s if c.isalpha()) >= 4
            and (len(s) >= 10 or " " in s)
        ):
            return s

        # config-driven heading patterns
        for rx in compiled_patterns:
            if rx.search(s):
                return s

        return None

    def _estimate_tokens(self, text: str) -> int:
        if self.app_settings.USE_TIKTOKEN_CHUNKING:
            try:
                import tiktoken

                encoding = tiktoken.get_encoding(self.app_settings.TIKTOKEN_ENCODING)
                return len(encoding.encode(text))
            except Exception:
                pass
        chars_per_token = self.app_settings.CHUNK_CHARS_PER_TOKEN or 4.0
        return max(int(len(text) / chars_per_token), 1)

    # -------------------- semantic chunking --------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Lightweight sentence splitter (no extra NLP dependency)."""
        # split on sentence-ending punctuation followed by whitespace + capital/quote,
        # while keeping the punctuation attached to the sentence.
        raw = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\"'(])", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 0]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def process_semantic_splitter(
        self,
        texts: List[str],
        metadatas: List[dict],
        embeddings,
        breakpoint_percentile: float = 95.0,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        Semantic chunking: splits each document into sentences, embeds them,
        and measures the distance (1 - cosine similarity) between each
        consecutive sentence pair. A "breakpoint" is inserted wherever that
        distance spikes above the given percentile — i.e. wherever the topic
        shifts noticeably — so each resulting chunk stays on one coherent
        topic instead of being cut at an arbitrary character count.

        `embeddings` must expose `.embed_documents(list[str]) -> list[list[float]]`
        (the standard LangChain Embeddings interface — works with OpenAI,
        HuggingFace, Cohere, Voyage, etc. wrappers).
        """
        chunks = []

        for text, metadata in zip(texts, metadatas):
            sentences = self._split_sentences(text)

            if len(sentences) <= 1:
                if sentences:
                    chunks.append(
                        Document(page_content=sentences[0], metadata=dict(metadata))
                    )
                continue

            sentence_embeddings = embeddings.embed_documents(sentences)

            # distance between each consecutive pair of sentences
            distances = [
                1 - self._cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
                for i in range(len(sentence_embeddings) - 1)
            ]

            # percentile threshold: distances above this mark a topic shift
            sorted_distances = sorted(distances)
            idx = min(
                len(sorted_distances) - 1,
                int(len(sorted_distances) * (breakpoint_percentile / 100.0)),
            )
            threshold = sorted_distances[idx] if sorted_distances else 0.0

            current_sentences = [sentences[0]]
            current_len = len(sentences[0])

            for i, distance in enumerate(distances):
                next_sentence = sentences[i + 1]
                is_breakpoint = distance > threshold
                would_exceed_max = current_len + len(next_sentence) > max_chunk_size
                has_min_size = current_len >= min_chunk_size

                if (is_breakpoint and has_min_size) or would_exceed_max:
                    chunks.append(
                        Document(
                            page_content=" ".join(current_sentences).strip(),
                            metadata=dict(metadata),
                        )
                    )
                    current_sentences = [next_sentence]
                    current_len = len(next_sentence)
                else:
                    current_sentences.append(next_sentence)
                    current_len += len(next_sentence)

            if current_sentences:
                chunks.append(
                    Document(
                        page_content=" ".join(current_sentences).strip(),
                        metadata=dict(metadata),
                    )
                )

        return chunks