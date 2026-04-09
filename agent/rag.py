"""FAISS-backed retrieval over curated grid-management guidelines."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from agent.schemas import RetrievedGuideline

KNOWLEDGE_DIR = Path("knowledge")
INDEX_PATH = Path("models/rag_index.faiss")
CORPUS_PATH = Path("models/rag_corpus.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]")


def _parse_markdown(path: Path) -> list[tuple[str, str]]:
    """Split a knowledge markdown file into (passage, source) tuples.

    Passages are the text between H2 headings; the source is pulled from the
    [Source: ...] trailer, which every passage in our corpus carries.
    """
    text = path.read_text()
    out: list[tuple[str, str]] = []
    sections = re.split(r"\n(?=##\s+)", text)
    for section in sections:
        if not section.strip().startswith("##"):
            continue
        passage = section.strip()
        m = _SOURCE_RE.search(passage)
        source = m.group(1).strip() if m else path.stem
        # Strip the [Source: ...] tag from the passage itself to keep embeddings clean
        clean = _SOURCE_RE.sub("", passage).strip()
        out.append((clean, source))
    return out


def _collect_corpus() -> list[dict]:
    items: list[dict] = []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for passage, source in _parse_markdown(md):
            items.append({"text": passage, "source": source, "file": md.name})
    return items


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def build_index(force: bool = False) -> None:
    """Build and persist the FAISS index + corpus manifest."""
    import faiss

    if INDEX_PATH.exists() and CORPUS_PATH.exists() and not force:
        return

    corpus = _collect_corpus()
    if not corpus:
        raise RuntimeError(f"No knowledge files found under {KNOWLEDGE_DIR}/")

    embedder = _get_embedder()
    vectors = embedder.encode([c["text"] for c in corpus], normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine via normalized inner product
    index.add(vectors)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))


@lru_cache(maxsize=1)
def _load_index():
    import faiss

    build_index(force=False)
    index = faiss.read_index(str(INDEX_PATH))
    corpus = json.loads(CORPUS_PATH.read_text())
    return index, corpus


def retrieve_guidelines(query: str, k: int = 4) -> list[RetrievedGuideline]:
    """Return the top-k passages most relevant to the query."""
    index, corpus = _load_index()
    embedder = _get_embedder()
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, idx = index.search(q, k)
    results: list[RetrievedGuideline] = []
    for rank in range(len(idx[0])):
        i = int(idx[0][rank])
        if i < 0:
            continue
        entry = corpus[i]
        results.append(
            RetrievedGuideline(
                text=entry["text"],
                source=entry["source"],
                score=float(scores[0][rank]),
            )
        )
    return results
