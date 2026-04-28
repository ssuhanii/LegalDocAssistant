"""
reranker.py
Cross-encoder reranker: takes the hybrid retriever's candidate chunks and
re-scores them using a more powerful cross-encoder model for higher precision.
"""

from langchain.schema import Document
from sentence_transformers import CrossEncoder

# ── Config ───────────────────────────────────────────────────────────────────
# This model is small (~67 MB) and works well for legal/factual text.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N          = 3   # final chunks passed to the LLM after reranking


class Reranker:
    """Wraps a sentence-transformers CrossEncoder to rerank retrieved chunks."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"  ⚙️  Loading reranker: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int = TOP_N,
    ) -> list[Document]:
        """
        Score each (query, chunk) pair and return the top_n highest-scoring docs.
        If fewer docs than top_n are supplied, all are returned as-is.
        """
        if not docs:
            return []

        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top    = scored[:top_n]

        # Attach rerank score as metadata so the UI can show it if desired
        results = []
        for score, doc in top:
            doc.metadata["rerank_score"] = round(float(score), 4)
            results.append(doc)

        return results


# ── Module-level singleton ────────────────────────────────────────────────────
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
