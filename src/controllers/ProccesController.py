from typing import List, Optional
from dataclasses import dataclass, field
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
        method: str = "recursive",  # "recursive" | "simple" | "semantic"
        embeddings=None,  # required when method="semantic"; any object with .embed_documents(List[str]) -> List[List[float]]
        semantic_breakpoint_percentile: float = 95.0,
        semantic_min_chunk_size: int = 0,
    ):
        """
        Chunk loaded documents.

        method="recursive" (default): LangChain's RecursiveCharacterTextSplitter —
            sentence/paragraph-aware, real overlap. Good general-purpose default.
        method="simple": the custom line-based fallback splitter.
        method="semantic": groups sentences by embedding similarity so each chunk
            stays topically coherent, instead of cutting at a fixed size. Costs an
            embedding call per document at ingest time. Requires `embeddings`.
        """
        file_content_texts = [rec.page_content for rec in file_content]
        file_content_metadata = [
            {**rec.metadata, "file_id": file_id} for rec in file_content
        ]

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