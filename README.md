# RAG From Scratch

A privacy-first, fully local Retrieval-Augmented Generation (RAG) pipeline. Every stage — document ingestion, embedding, vector storage, and LLM generation — runs on your own machine. Nothing leaves your server.

## Architecture

The pipeline is split into three phases, each exposed as a CLI command:

| Phase | Command | What it does |
|-------|---------|--------------|
| 1 — Ingest | `ingest` | Load documents and split them into semantic chunks |
| 2 — Embed & Store | `embed` | Embed chunks via Ollama and store them in PostgreSQL + pgvector |
| 2 — Retrieve | `retrieve` | Run a cosine-similarity search against stored chunks |
| 3 — Query | `query` | Full RAG flow: embed → retrieve → prompt → generate an answer |

### Supported File Types

| Extension | Parser |
|-----------|--------|
| `.pdf` | Docling → Markdown |
| `.docx` | Docling → Markdown |
| `.html` / `.htm` | BeautifulSoup + lxml |
| `.txt` / `.md` | Plain text (UTF-8, latin-1 fallback) |
| HTTP/HTTPS URLs | HTML fetch + BeautifulSoup |

---

## Prerequisites

| Dependency | Required for | Default URL |
|------------|-------------|-------------|
| **Python ≥ 3.12** | Everything | — |
| **[uv](https://docs.astral.sh/uv/)** | Package management | — |
| **[Ollama](https://ollama.com/)** | Embeddings (`embed`, `retrieve`, `query`) | `http://localhost:11434` |
| **PostgreSQL 17 + [pgvector](https://github.com/pgvector/pgvector)** | Vector storage (`embed`, `retrieve`, `query`) | `localhost:5432` |
| **[LM Studio](https://lmstudio.ai/)** _or_ Ollama | LLM generation (`query`) | `http://localhost:1234` |

> **Note:** The `ingest` command has no external service dependencies — it only requires Python and the installed packages.

### Pull the Ollama embedding model

```bash
ollama pull embeddinggema:latest
```

### Create the PostgreSQL database

```bash
createdb rag
# or via psql:
psql -c "CREATE DATABASE rag;"
```

The pipeline auto-creates the `vector` extension, `chunks` table, and HNSW index on first `embed` run.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/BrianYue064/RAGFromScratch.git
cd RAGFromScratch

# Create a virtual environment and install dependencies
uv sync

# Install dev dependencies (pytest, pytest-docker)
uv sync --extra dev

# (Optional) Copy and edit the environment config
cp .env.example .env
```

All configuration is read from environment variables prefixed with `RAG_`. See [`.env.example`](.env.example) for the full list with defaults.

---

## CLI Usage

Run commands through `uv run`:

### Ingest — load and chunk documents (Phase 1)

```bash
# Single file
uv run python main.py ingest path/to/document.pdf

# Multiple files
uv run python main.py ingest file1.txt file2.html https://example.com/page

# Entire folder (one level deep, supported extensions only)
uv run python main.py ingest path/to/folder/

# Custom chunk size and overlap
uv run python main.py ingest document.pdf --chunk-size 256 --overlap 25
```

### Embed — ingest, embed, and store in pgvector (Phase 2)

```bash
# Requires: Ollama running + PostgreSQL + pgvector
uv run python main.py embed path/to/document.pdf
uv run python main.py embed path/to/folder/
uv run python main.py embed file1.pdf file2.txt --chunk-size 1024
```

### Retrieve — query the vector store (Phase 2)

```bash
# Requires: Ollama running + PostgreSQL + pgvector
uv run python main.py retrieve "What is retrieval-augmented generation?"
uv run python main.py retrieve "search query" --top-k 10
```

### Query — full RAG question answering (Phase 3)

```bash
# Requires: Ollama running + PostgreSQL + pgvector + LLM provider
uv run python main.py query "What are the key findings in the report?"
uv run python main.py query "Summarize the main points" --top-k 3
```

---

## Configuration

All settings are controlled via `RAG_`-prefixed environment variables. Set them in your shell, in a `.env` file, or export them before running:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_CHUNK_SIZE` | `512` | Target token count per chunk |
| `RAG_OVERLAP` | `50` | Overlap tokens between consecutive chunks |
| `RAG_ENCODING` | `cl100k_base` | tiktoken encoding name |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model name |
| `RAG_EMBEDDING_DIM` | `768` | Expected embedding vector dimension (must match model) |
| `RAG_DB_URL` | `postgresql://postgres:postgres@localhost:5432/rag` | PostgreSQL connection string |
| `RAG_LLM_PROVIDER` | `lmstudio` | LLM backend: `"ollama"` or `"lmstudio"` |
| `RAG_LLM_MODEL` | `qwen3.5-0.8b` | LLM model name (must be loaded in the provider) |
| `RAG_LMSTUDIO_URL` | `http://localhost:1234` | LM Studio server URL |
| `RAG_LLM_TEMPERATURE` | `0.7` | Generation temperature |
| `RAG_LLM_MAX_TOKENS` | `1024` | Maximum tokens to generate |
| `RAG_TOP_K` | `5` | Default number of chunks returned by retrieval |

---

## Running Tests

The test suite has **54 tests** across 8 test files: 43 unit tests (no external services) and 11 integration tests (require Docker).

### Unit tests only (no Docker required)

```bash
uv run pytest rag_pipeline/tests/ -m "not integration" -v
```

### Integration tests (require Docker)

Integration tests use `pytest-docker` to spin up a PostgreSQL + pgvector container automatically.

```bash
# Make sure Docker is running, then:
uv run pytest rag_pipeline/tests/ -m integration -v
```

### All tests

```bash
uv run pytest rag_pipeline/tests/ -v
```

### Test breakdown

| Test file | Count | Type | Covers |
|-----------|-------|------|--------|
| `test_loader.py` | 8 | Unit | File/URL loading, metadata, error handling |
| `test_chunker.py` | 9 | Unit | Chunking, token limits, overlap, edge cases |
| `test_embedder.py` | 8 | Unit | Ollama embed client (HTTP mocked) |
| `test_llm_client.py` | 9 | Unit | Ollama + LM Studio generate (HTTP mocked) |
| `test_prompt_builder.py` | 5 | Unit | Prompt template assembly |
| `test_query.py` | 4 | Unit | RAG query orchestrator (fully mocked) |
| `test_pg_store.py` | 9 | Integration | pgvector CRUD + similarity search |
| `test_end_to_end.py` | 2 | Integration | Full pipeline: load → chunk → embed → store → retrieve |

---

## Project Structure

```
RAGFromScratch/
├── main.py                             # CLI entry point (argparse, 4 commands)
├── pyproject.toml                      # Dependencies, build config, pytest markers
├── .env.example                        # Environment variable reference
├── .python-version                     # Python 3.12
├── uv.lock                            # uv lockfile
│
└── rag_pipeline/                       # Main package
    ├── config.py                       # Config dataclass + load_config()
    ├── models.py                       # Shared exception hierarchy
    ├── conftest.py                     # Pytest path setup + Docker fixture
    │
    ├── ingestion/                      # Phase 1: Document loading & chunking
    │   ├── loader.py                   #   File/URL loading dispatcher
    │   ├── chunker.py                  #   Sentence-boundary chunking (NLTK + tiktoken)
    │   ├── resolver.py                 #   Multi-path / folder resolution
    │   ├── models.py                   #   Document & Chunk dataclasses
    │   └── parsers/
    │       ├── html_parser.py          #   BeautifulSoup HTML parser (files + URLs)
    │       └── text_parser.py          #   Plain text / markdown parser
    │
    ├── vectorstore/                    # Phase 2: Embedding & storage
    │   ├── embedder.py                 #   Ollama /api/embed client
    │   ├── pg_store.py                 #   PostgreSQL + pgvector store
    │   └── models.py                   #   EmbeddedChunk dataclass
    │
    ├── generation/                     # Phase 3: LLM generation
    │   ├── llm_client.py               #   Ollama + LM Studio HTTP clients
    │   ├── prompt_builder.py           #   RAG prompt template assembly
    │   └── query.py                    #   Full RAG orchestrator
    │
    └── tests/
        ├── docker-compose.yml          #   pgvector:pg17 for integration tests
        ├── fixtures/                   #   Sample files (.txt, .pdf, .html, .md, .docx)
        ├── test_loader.py
        ├── test_chunker.py
        ├── test_embedder.py
        ├── test_pg_store.py
        ├── test_end_to_end.py
        ├── test_llm_client.py
        ├── test_prompt_builder.py
        └── test_query.py
```

---

## Design Decisions

**Docling over unstructured.io** — Docling is a local, privacy-preserving library that converts PDFs and DOCX files to clean markdown without sending data to external APIs. It produces well-structured output ideal for chunking.

**tiktoken as measurement, not processing** — tiktoken (cl100k_base encoding) is used only to count tokens and measure chunk sizes. It is not used to generate embeddings or modify text.

**Sentence-boundary chunking** — Splitting on sentence boundaries (via NLTK `sent_tokenize`) preserves semantic coherence better than arbitrary fixed-size splitting. The chunker accumulates sentences until the token limit is reached, then carries overlapping tokens to the next chunk to maintain context across boundaries.

**Ollama for embeddings** — All embedding calls go through Ollama's local `/api/embed` endpoint. The default model (`embeddinggemma:latest`, 768 dimensions) runs entirely on the local machine.

**PostgreSQL + pgvector for storage** — Embeddings are stored in PostgreSQL with the pgvector extension, using cosine similarity (`<=>` operator) for retrieval and an HNSW index for fast approximate nearest neighbor search.

**Dual LLM provider support** — The generation layer supports both Ollama (`/api/generate`) and LM Studio (OpenAI-compatible `/v1/chat/completions`), selectable via the `RAG_LLM_PROVIDER` environment variable.

---

## Known Limitations

- **Folder scanning is one level deep** — `resolve_paths()` only finds files that are direct children of a given directory; subdirectories are skipped.
- **Embeddings are sequential** — Chunks are embedded one at a time via individual HTTP calls to Ollama. There is no batch embedding. This is slow for large document sets but avoids overloading limited local hardware.
- **No duplicate detection** — Re-ingesting the same document creates duplicate chunks in the vector store. There is no content-hash or source-based deduplication.
- **Two LLM providers only** — Only `"ollama"` and `"lmstudio"` are supported. There is no plugin system for adding providers.
- **No streaming output** — LLM responses are returned in full (no progressive/streaming output).
- **Text encoding** — Plain text files are read as UTF-8 with a latin-1 fallback. Other encodings (UTF-16, Shift-JIS, etc.) are not handled.
- **HTML extraction** — The HTML parser focuses on `<p>` tags inside `<main>`, `<article>`, or `<body>`. Content in lists, tables, or headings outside of `<p>` elements may be lost.
- **Hard-coded timeouts** — Embedding and LLM requests time out at 120 seconds; URL fetching times out at 10 seconds. These are not configurable via environment variables.