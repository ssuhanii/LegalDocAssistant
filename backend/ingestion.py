"""
ingestion.py
Loads legal PDF documents, splits them into chunks, embeds them,
and stores them in a FAISS vector index alongside a BM25 index.
"""

import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DOCS_DIR       = BASE_DIR / "data" / "documents"
FAISS_INDEX    = BASE_DIR / "data" / "faiss_index"
BM25_INDEX     = BASE_DIR / "data" / "bm25_index.pkl"
CHUNKS_CACHE   = BASE_DIR / "data" / "chunks.pkl"

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a HuggingFace sentence-transformer embedding model.

    all-MiniLM-L6-v2 is compact (~90 MB), fast, and requires no extra
    dependencies — ideal for local legal document search.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: Path = DOCS_DIR) -> list:
    """Walk docs_dir and load every PDF found."""
    pdf_files = list(docs_dir.glob("**/*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{docs_dir}'. "
            "Drop your legal PDFs into data/documents/ and re-run."
        )

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  📄 Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Attach the source filename as metadata
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        all_docs.extend(docs)

    print(f"  ✅ Loaded {len(all_docs)} pages from {len(pdf_files)} PDF(s).")
    return all_docs


def split_documents(docs: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✅ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_faiss_index(chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Embed chunks and persist a FAISS vector store."""
    print("  🔢 Building FAISS index — this may take a minute…")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_INDEX.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX))
    print(f"  ✅ FAISS index saved → {FAISS_INDEX}")
    return vectorstore


def build_bm25_index(chunks: list) -> BM25Okapi:
    """Build a BM25 keyword index from chunk texts."""
    print("  🔤 Building BM25 index…")
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)

    # Persist both the BM25 model and the raw chunks (needed at query time)
    with open(BM25_INDEX, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  ✅ BM25 index saved  → {BM25_INDEX}")
    return bm25


def ingest(docs_dir: Path = DOCS_DIR) -> None:
    """Full ingestion pipeline: load → split → embed → index."""
    print("\n🚀 Starting ingestion pipeline…\n")

    docs   = load_documents(docs_dir)
    chunks = split_documents(docs)

    embeddings = _get_embedding_model()

    build_faiss_index(chunks, embeddings)
    build_bm25_index(chunks)

    print("\n🎉 Ingestion complete! Indexes are ready for querying.\n")


if __name__ == "__main__":
    ingest()
