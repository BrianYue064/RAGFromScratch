"""FastAPI application for the RAG pipeline."""

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse

from rag_pipeline.config import Config, load_config
from rag_pipeline.ingestion import chunk, load
from rag_pipeline.models import (
    EmbeddingError,
    GenerationError,
    IngestionError,
    RAGPipelineError,
    VectorStoreError,
)
from rag_pipeline.vectorstore import PgVectorStore, embed_chunks
from rag_pipeline.generation import rag_query, rag_query_stream

from .schemas import (
    ChunksListResponse,
    DeleteResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    config = load_config()
    app.state.config = config

    logger.info("Initializing vector store connection...")
    store = PgVectorStore(config)
    try:
        store.initialize()
        app.state.store = store
        logger.info("API Startup complete.")
        yield
    finally:
        logger.info("API Shutdown: closing vector store connection.")
        if hasattr(app.state, "store"):
            app.state.store.close()


app = FastAPI(
    title="RAG Pipeline API",
    description="Privacy-first, fully local RAG pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RAGPipelineError)
async def rag_exception_handler(request: Request, exc: RAGPipelineError):
    """Global exception handler for pipeline errors."""
    status_code = 500
    if isinstance(exc, (EmbeddingError, IngestionError)):
        status_code = 503  # Service Unavailable (Ollama down, etc.)
    elif isinstance(exc, GenerationError):
        status_code = 502  # Bad Gateway (LLM down)

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__, detail=str(exc)
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect to OpenAPI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Readiness check for database and Ollama."""
    config: Config = app.state.config
    store: PgVectorStore = app.state.store

    db_status = "connected"
    try:
        store.count()
    except Exception as e:
        db_status = f"error: {e}"

    ollama_status = "reachable"
    try:
        # Simple readiness check for Ollama
        with httpx.Client() as client:
            res = client.get(f"{config.ollama_url}/api/tags", timeout=5)
            res.raise_for_status()
    except Exception as e:
        ollama_status = f"error: {e}"

    status = "healthy" if db_status == "connected" and ollama_status == "reachable" else "degraded"

    return HealthResponse(
        status=status,
        database=db_status,
        ollama=ollama_status,
    )


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """Full RAG query: embed → retrieve → generate."""
    config: Config = app.state.config

    if not req.stream:
        result = rag_query(
            req.question,
            config,
            top_k=req.top_k,
            model=req.model,
            provider=req.provider,
        )
        return QueryResponse(**result)

    sources, token_gen = rag_query_stream(
        req.question,
        config,
        top_k=req.top_k,
        model=req.model,
        provider=req.provider,
    )

    async def sse_generator():
        # Yield tokens in OpenAI-compatible SSE format
        for token in token_gen:
            yield {
                "data": json.dumps(
                    {
                        "choices": [{"delta": {"content": token}, "finish_reason": None}]
                    }
                )
            }
        
        # Final event with sources
        yield {
            "data": json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "sources": sources,
                }
            )
        }
        
        # Terminator
        yield {"data": "[DONE]"}

    return EventSourceResponse(sse_generator())


@app.post("/embed", response_model=EmbedResponse)
async def embed_endpoint(req: EmbedRequest):
    """Ingest, chunk, embed, and store a document."""
    config: Config = app.state.config
    store: PgVectorStore = app.state.store

    chunk_size = req.chunk_size or config.chunk_size
    overlap = req.overlap or config.overlap

    doc = load(req.path)
    chunks_list = chunk(doc, chunk_size=chunk_size, overlap=overlap)
    
    if not chunks_list:
        return EmbedResponse(
            source=doc.source,
            chunks_stored=0,
            embedding_model=config.embedding_model,
            embedding_dim=config.embedding_dim,
        )

    embedded = embed_chunks(chunks_list, config)
    count = store.insert(embedded)

    return EmbedResponse(
        source=doc.source,
        chunks_stored=count,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
    )


@app.get("/chunks", response_model=ChunksListResponse)
async def list_chunks(source: str | None = None, limit: int = 50):
    """List stored chunks, optionally filtered by source."""
    store: PgVectorStore = app.state.store
    
    # We query PgVectorStore manually here
    sql = "SELECT chunk_id, text, source, chunk_index, token_count FROM chunks"
    params = []
    
    if source:
        sql += " WHERE source = %s"
        params.append(source)
        
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    try:
        with store._conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            # Convert UUIDs to strings
            for row in rows:
                row["chunk_id"] = str(row["chunk_id"])
            
            total = store.count()
            
            return ChunksListResponse(chunks=rows, total=total)
    except Exception as e:
        raise VectorStoreError(f"Failed to list chunks: {e}") from e


@app.delete("/chunks/{source:path}", response_model=DeleteResponse)
async def delete_chunks(source: str):
    """Delete all chunks for a given source."""
    store: PgVectorStore = app.state.store
    count = store.delete_by_source(source)
    return DeleteResponse(source=source, chunks_deleted=count)
