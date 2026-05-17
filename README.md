# RAG Pipeline - Data Ingestion and Chunking Module

This module handles Phase 1 of a privacy-first, fully local RAG (Retrieval-Augmented Generation) pipeline. It is responsible for loading documents from various file formats and splitting them into semantic chunks suitable for vector embedding. All data processing happens locally on the machine—nothing leaves the server.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the NLTK sentence tokenizer (one-time setup):

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Usage

```python
from ingestion import load, chunk

doc = load("document.pdf")
chunks = chunk(doc, chunk_size=512, overlap=50)

print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0].text[:200]}...")
```

## Design Decisions

**Docling over unstructured.io**: Docling is a local, privacy-preserving library that converts PDFs and DOCX files to clean markdown without sending data to external APIs. It produces well-structured output ideal for chunking.

**tiktoken as measurement, not processing**: tiktoken is used only to count tokens and measure chunk sizes. It is not used to generate embeddings or modify text.

**Sentence-boundary chunking**: Splitting on sentence boundaries (via NLTK's `sent_tokenize`) preserves semantic coherence better than arbitrary fixed-size splitting. The chunker accumulates sentences until the token limit is reached, then carries overlapping tokens to the next chunk to maintain context across chunk boundaries.