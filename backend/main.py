"""
main.py
FastAPI application — exposes the RAG pipeline as HTTP endpoints.

Endpoints
---------
POST /query          – Ask a question against the indexed documents
POST /ingest         – Re-run the ingestion pipeline (hot reload)
GET  /health         – Liveness check
GET  /sources        – List all ingested PDF filenames
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Make project root importable when running with uvicorn ──────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag_pipeline import get_pipeline
from backend.ingestion    import ingest, DOCS_DIR, CHUNKS_CACHE

load_dotenv()

# ── Lifespan: warm up pipeline on startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the RAG pipeline so the first request isn't slow."""
    if CHUNKS_CACHE.exists():
        print("Warming up RAG pipeline…")
        get_pipeline()
        print("Pipeline ready.")
    else:
        print("No index found — run POST /ingest to index your documents.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LegalDocAssistant API",
    description="Hybrid RAG over legal PDF documents using local Ollama LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    role:    str
    content: str

class QueryRequest(BaseModel):
    question:     str             = Field(..., min_length=3, example="What are the termination clauses?")
    retrieve_k:   int             = Field(10, ge=1, le=20)
    rerank_n:     int             = Field(3,  ge=1, le=10)
    chat_history: list[ChatTurn]  = Field(default_factory=list)

class SourceDoc(BaseModel):
    source:       str
    page:         int | str
    snippet:      str
    rerank_score: float | None = None

class LatencyBreakdown(BaseModel):
    retrieval_s: float
    rerank_s:    float
    llm_s:       float
    total_s:     float

# ── THIS IS THE UPDATED QueryResponse — adds latency field ───────────────────
class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceDoc]
    latency: LatencyBreakdown        # ← new

class IngestResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status:      str
    index_ready: bool
    docs_dir:    str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    return {
        "status":      "ok",
        "index_ready": CHUNKS_CACHE.exists(),
        "docs_dir":    str(DOCS_DIR),
    }


@app.get("/sources", tags=["Utility"])
def list_sources():
    """Return the names of all PDFs currently in data/documents/."""
    pdfs = [p.name for p in DOCS_DIR.glob("**/*.pdf")]
    return {"documents": pdfs, "count": len(pdfs)}


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def run_ingestion():
    """
    Trigger the full ingestion pipeline.
    Call this whenever you add new PDFs to data/documents/.
    """
    try:
        ingest(DOCS_DIR)
        # Reset the singleton so next query uses fresh indexes
        import backend.rag_pipeline as rp
        rp._pipeline = None
        return {"message": "Ingestion complete. Indexes are up to date."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    """Ask a question — returns answer, citations, and latency breakdown."""
    if not CHUNKS_CACHE.exists():
        raise HTTPException(
            status_code=503,
            detail="No index found. POST to /ingest first.",
        )
    try:
        pipeline = get_pipeline()
        result   = pipeline.query(
            question     = req.question,
            retrieve_k   = req.retrieve_k,
            rerank_n     = req.rerank_n,
            chat_history = [t.model_dump() for t in req.chat_history],
        )
        sources = [
            SourceDoc(
                source       = doc.metadata.get("source", "unknown"),
                page         = doc.metadata.get("page", "?"),
                snippet      = doc.page_content[:300],
                rerank_score = doc.metadata.get("rerank_score"),
            )
            for doc in result["sources"]
        ]
        return QueryResponse(
            answer  = result["answer"],
            sources = sources,
            latency = result["latency"],   # ← new
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))