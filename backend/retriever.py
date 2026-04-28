"""
retriever.py
Hybrid retriever: combines FAISS dense retrieval with BM25 sparse retrieval
using Reciprocal Rank Fusion (RRF) to produce a single ranked result list.
"""

import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
FAISS_INDEX  = BASE_DIR / "data" / "faiss_index"
BM25_INDEX   = BASE_DIR / "data" / "bm25_index.pkl"
CHUNKS_CACHE = BASE_DIR / "data" / "chunks.pkl"

# ── Config ───────────────────────────────────────────────────────────────────
TOP_K       = 10   # candidates per retriever before fusion
FINAL_TOP_K = 5    # results returned after fusion
RRF_K       = 60   # RRF constant (higher → smooths rank differences)


def _get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


class HybridRetriever:
    """Loads persisted FAISS + BM25 indexes and performs hybrid retrieval."""

    def __init__(self):
        self._faiss: FAISS | None = None
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Document] | None = None
        self._embeddings: HuggingFaceEmbeddings | None = None

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load_faiss(self) -> FAISS:
        if self._faiss is None:
            if not FAISS_INDEX.exists():
                raise RuntimeError(
                    "FAISS index not found. Run ingestion.py first."
                )
            self._embeddings = self._embeddings or _get_embedding_model()
            self._faiss = FAISS.load_local(
                str(FAISS_INDEX),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._faiss

    def _load_bm25(self) -> tuple[BM25Okapi, list[Document]]:
        if self._bm25 is None:
            if not BM25_INDEX.exists() or not CHUNKS_CACHE.exists():
                raise RuntimeError(
                    "BM25 index not found. Run ingestion.py first."
                )
            with open(BM25_INDEX, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(CHUNKS_CACHE, "rb") as f:
                self._chunks = pickle.load(f)
        return self._bm25, self._chunks

    # ── Individual retrievers ─────────────────────────────────────────────────

    def _faiss_search(self, query: str, k: int) -> list[Document]:
        faiss = self._load_faiss()
        return faiss.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        bm25, chunks = self._load_bm25()
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [chunks[i] for i in top_indices]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        lists: list[list[Document]],
        k: int = RRF_K,
        final_k: int = FINAL_TOP_K,
    ) -> list[Document]:
        """Merge multiple ranked lists using RRF scoring."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked_list in lists:
            for rank, doc in enumerate(ranked_list):
                # Use content hash as a dedup key
                key = doc.page_content[:200]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:final_k]]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[Document]:
        """Return the top-k most relevant chunks for a query."""
        faiss_results = self._faiss_search(query, k=TOP_K)
        bm25_results  = self._bm25_search(query, k=TOP_K)
        fused = self._rrf_fuse([faiss_results, bm25_results], final_k=top_k)
        return fused


# Module-level singleton — shared across requests
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
