"""
rag_pipeline.py
Orchestrates the full RAG flow:
 1. Hybrid retrieve   (FAISS + BM25 → RRF fusion)
 2. Rerank            (cross-encoder)
 3. Generate answer   (Ollama / llama3.2 via LangChain)
 4. Track latency     (retrieval / rerank / LLM / total)
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from backend.retriever import get_retriever
from backend.reranker  import get_reranker

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2")

# ── Prompt template ───────────────────────────────────────────────────────────
LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are an expert legal assistant specialising in Indian law.
Answer the user's question using the Context provided below AND the Conversation so far.
If the answer cannot be determined from the Context or the Conversation so far, say exactly:
"I could not find relevant information in the provided documents."

Be precise, cite clause numbers or section headings when available,
and never add factual information not present in the Context or the Conversation so far.

Context:
{context}

{chat_history}Question:
{question}

Answer:""",
)


def _format_context(docs: list[Document]) -> str:
    """Concatenate chunk texts with source labels for the prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] (Source: {source}, Page: {page})\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    """End-to-end Retrieve → Rerank → Generate pipeline with latency tracking."""

    def __init__(self):
        self._retriever = get_retriever()
        self._reranker  = get_reranker()
        self._llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=0.1,   # low temp → deterministic, factual answers
            num_predict=512,
        )
        self._chain = LEGAL_PROMPT | self._llm

    def query(
        self,
        question:     str,
        retrieve_k:   int = 10,
        rerank_n:     int = 3,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """
        Run the full pipeline for a user question.

        Parameters
        ----------
        question     : the current user question
        retrieve_k   : hybrid retrieval candidate count
        rerank_n     : chunks passed to the LLM after reranking
        chat_history : list of {"role": "user"|"assistant", "content": str}

        Returns
        -------
        dict with keys:
            answer   : str   – LLM-generated answer
            sources  : list  – the reranked Document objects used
            latency  : dict  – breakdown of time spent at each stage (seconds)
        """

        # ── Step 1: Hybrid retrieval ──────────────────────────────────────────
        t0 = time.perf_counter()
        candidates = self._retriever.retrieve(question, top_k=retrieve_k)
        retrieval_time = round(time.perf_counter() - t0, 3)

        # ── Step 2: Rerank ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        top_docs = self._reranker.rerank(question, candidates, top_n=rerank_n)
        rerank_time = round(time.perf_counter() - t1, 3)

        # ── Step 3: Format chat history for the prompt ────────────────────────
        history_text = ""
        if chat_history:
            lines = []
            for turn in chat_history:
                role = "User" if turn["role"] == "user" else "Assistant"
                lines.append(f"{role}: {turn['content']}")
            history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        # ── Step 4: Generate answer ───────────────────────────────────────────
        t2 = time.perf_counter()
        context = _format_context(top_docs)
        answer  = self._chain.invoke({
            "context":      context,
            "chat_history": history_text,
            "question":     question,
        })
        llm_time = round(time.perf_counter() - t2, 3)

        total_time = round(retrieval_time + rerank_time + llm_time, 3)

        # ── Log to console for debugging ──────────────────────────────────────
        print(
            f"\n⏱  Latency breakdown → "
            f"Retrieval: {retrieval_time}s | "
            f"Rerank: {rerank_time}s | "
            f"LLM: {llm_time}s | "
            f"Total: {total_time}s"
        )

        return {
            "answer":  answer.strip(),
            "sources": top_docs,
            "latency": {
                "retrieval_s": retrieval_time,
                "rerank_s":    rerank_time,
                "llm_s":       llm_time,
                "total_s":     total_time,
            },
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline