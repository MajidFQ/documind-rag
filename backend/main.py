"""
main.py — FastAPI application for DocuMind.

This file is the HTTP interface that sits in front of the entire RAG pipeline.
It exposes four endpoints:

  POST /upload
    Accepts a PDF or DOCX file plus a chunking strategy, runs the full
    ingestion pipeline (extract → chunk → embed → store), and returns how
    many chunks were stored. This is the "ingest a document" button.

  POST /query
    Accepts a question, collection name, and retrieval options. Runs the
    full RAG pipeline (retrieve → rerank → generate → faithfulness check)
    using model instances that were loaded ONCE at startup — not per request.
    Returns the answer, citations, and faithfulness score.

  GET /collections
    Lists all ChromaDB collections. The frontend uses this to populate a
    document picker so users can choose which document set to query.

  DELETE /collections/{collection_name}
    Deletes a collection. Useful for resetting during testing or letting
    users remove an uploaded document.

Startup:
  All heavy objects (EmbeddingModel, VectorStore, CrossEncoderReranker, Groq
  client) are loaded once inside the lifespan context manager and stored on
  app.state. Every request handler reads from app.state instead of creating
  new instances, keeping per-request latency low.
"""

import os
import pathlib
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

# Load .env before importing config (config reads env vars at import time)
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from .config import GROQ_API_KEY, CHROMA_COLLECTION_NAME, TOP_K
from .embeddings import EmbeddingModel, VectorStore, process_and_store
from .retrieval import CrossEncoderReranker
from .generation import answer_query

# ── Allowed upload extensions ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Lifespan — load all heavy objects once at startup, clean up on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: runs setup before the server accepts requests,
    and teardown after it stops.

    WHY lifespan instead of @app.on_event("startup"):
      The lifespan pattern is the current FastAPI recommendation. It keeps
      startup and shutdown logic together in one place, and it avoids the
      deprecation warning that on_event now emits.

    WHY load models here instead of per-request:
      EmbeddingModel and CrossEncoderReranker each take 1–3 seconds to load
      from disk. Loading them per-request would make every query feel slow.
      Loading once and sharing via app.state costs nothing extra per request.
    """
    print("DocuMind startup: loading models ...")

    app.state.embedding_model = EmbeddingModel()
    app.state.vector_store    = VectorStore()
    app.state.reranker        = CrossEncoderReranker()
    app.state.groq_client     = Groq(api_key=GROQ_API_KEY)

    print("DocuMind startup: all models ready. Server accepting requests.")

    yield  # server runs here

    # Shutdown — nothing to explicitly close for these objects, but the hook
    # is here so future cleanup (connection pools, etc.) has a place to go.
    print("DocuMind shutdown: cleaning up.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DocuMind API",
    description="RAG-powered document Q&A backend.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow a Streamlit frontend (or any local dev client) on any port
# to call this API. In production you'd restrict allow_origins to your
# actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper — validate upload file type
# ---------------------------------------------------------------------------

def validate_file_extension(filename: str) -> str:
    """
    Return the lowercase extension if allowed, otherwise raise HTTP 400.

    Centralising this check means both the extension check and the error
    message live in one place instead of being repeated across routes.

    Args:
        filename: The original filename from the upload.

    Returns:
        The lowercase extension string (e.g. '.pdf').

    Raises:
        HTTPException 400: If the extension is not in ALLOWED_EXTENSIONS.
    """
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"DocuMind accepts: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )
    return ext


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post("/upload", summary="Upload and ingest a document")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF or DOCX file to ingest"),
    chunking_method: str = Form(
        default="recursive",
        description="Chunking strategy: 'recursive' or 'fixed_size'",
    ),
    collection_name: str = Form(
        default=CHROMA_COLLECTION_NAME,
        description="ChromaDB collection to store chunks in",
    ),
):
    """
    Accept a document upload, run the full ingestion pipeline, and persist
    the resulting chunks and embeddings to ChromaDB.

    The file is saved to a temporary path so process_and_store() can read it
    normally. The temp file is deleted after ingestion regardless of success
    or failure.

    Returns:
        JSON with chunks_stored (int), collection_name (str), filename (str).
    """
    validate_file_extension(file.filename)

    if chunking_method not in ("recursive", "fixed_size"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown chunking_method '{chunking_method}'. "
                "Choose 'recursive' or 'fixed_size'."
            ),
        )

    # Write upload to a named temp file so process_and_store() can open it
    # by path (the ingestion layer expects a file path, not a file object)
    suffix = pathlib.Path(file.filename).suffix.lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        chunks_stored = process_and_store(
            file_path=tmp_path,
            chunking_method=chunking_method,
            collection_name=collection_name,
            original_filename=file.filename,
        )

    except HTTPException:
        raise  # re-raise validation errors unchanged
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed for '{file.filename}': {error}",
        )
    finally:
        # Always clean up the temp file — even if ingestion raised an error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "chunks_stored": chunks_stored,
        "collection_name": collection_name,
        "chunking_method": chunking_method,
    }


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", summary="Ask a question against an ingested document collection")
async def query_documents(request: Request, body: dict):
    """
    Run the full RAG pipeline for a user question and return a grounded answer.

    Reads model instances from app.state (loaded once at startup) so no
    model is reloaded per request.

    Expected JSON body:
        {
          "query":           "What is RAG?",          // required
          "collection_name": "documind",              // optional
          "top_k":           5,                       // optional
          "rerank":          true                     // optional
        }

    Returns:
        JSON with answer, sources_cited, faithfulness_score,
        faithfulness_explanation, and retrieved_chunks.
    """
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="'query' field is required and must not be empty.",
        )

    collection_name = body.get("collection_name", CHROMA_COLLECTION_NAME)
    top_k           = int(body.get("top_k", TOP_K))
    rerank          = bool(body.get("rerank", True))

    try:
        result = answer_query(
            query=query,
            collection_name=collection_name,
            groq_client=request.app.state.groq_client,
            top_k=top_k,
            rerank=rerank,
            embedding_model=request.app.state.embedding_model,
            vector_store=request.app.state.vector_store,
            reranker=request.app.state.reranker,
        )
    except ValueError as error:
        # ValueError from retrieval = empty collection or bad input
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Query pipeline failed: {error}",
        )

    # Slim down sources_cited for the JSON response — drop raw embedding
    # vectors (not stored anyway) but keep all metadata fields
    slim_sources = [
        {
            "source":      src.get("source"),
            "page_number": src.get("page_number"),
            "chunk_id":    src.get("chunk_id"),
            "text":        src.get("text"),
        }
        for src in result["sources_cited"]
    ]

    slim_chunks = [
        {
            "source":       c.get("source"),
            "page_number":  c.get("page_number"),
            "chunk_id":     c.get("chunk_id"),
            "relevance":    c.get("relevance"),
            "rerank_score": c.get("rerank_score"),
            "text":         c.get("text"),
        }
        for c in result["retrieved_chunks"]
    ]

    return {
        "query":                    query,
        "answer":                   result["answer"],
        "sources_cited":            slim_sources,
        "faithfulness_score":       result["faithfulness_score"],
        "faithfulness_explanation": result["faithfulness_explanation"],
        "retrieved_chunks":         slim_chunks,
    }


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------

@app.get("/collections", summary="List all ChromaDB collections")
async def list_collections(request: Request):
    """
    Return the names and document counts of all existing ChromaDB collections.

    The frontend uses this to populate a dropdown so users can choose which
    ingested document set to query against.

    Returns:
        JSON with a list of { name, count } objects.
    """
    try:
        client = request.app.state.vector_store.client
        collections = client.list_collections()
        result = []
        for col in collections:
            # Fetch the actual collection object to read its document count
            col_obj = client.get_collection(col.name)
            result.append({
                "name":  col.name,
                "count": col_obj.count(),
            })
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {error}",
        )

    return {"collections": result}


# ---------------------------------------------------------------------------
# DELETE /collections/{collection_name}
# ---------------------------------------------------------------------------

@app.delete(
    "/collections/{collection_name}",
    summary="Delete a ChromaDB collection",
)
async def delete_collection(collection_name: str, request: Request):
    """
    Delete a named ChromaDB collection and all its stored chunks.

    Used for resetting during testing or allowing users to remove a document
    they no longer want in the system.

    Returns:
        JSON confirming the deleted collection name.

    Raises:
        HTTP 404 if the collection does not exist.
    """
    client = request.app.state.vector_store.client

    # Check existence before attempting delete so we return 404, not 500
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found.",
        )

    try:
        client.delete_collection(collection_name)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection '{collection_name}': {error}",
        )

    return {"deleted": collection_name}
