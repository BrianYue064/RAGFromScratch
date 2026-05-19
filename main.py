"""CLI entry point for the RAG pipeline."""

import argparse
import sys

from rag_pipeline.config import load_config
from rag_pipeline.ingestion import chunk, load


def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Privacy-first, fully local RAG pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- ingest (Phase 1) ---
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest and chunk a document"
    )
    ingest_parser.add_argument(
        "path", type=str, help="File path or URL to ingest"
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Target token count per chunk",
    )
    ingest_parser.add_argument(
        "--overlap", type=int, default=None,
        help="Overlap tokens between chunks",
    )

    # --- embed (Phase 2) ---
    embed_parser = subparsers.add_parser(
        "embed", help="Ingest, embed, and store a document"
    )
    embed_parser.add_argument(
        "path", type=str, help="File path or URL to ingest"
    )
    embed_parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Target token count per chunk",
    )
    embed_parser.add_argument(
        "--overlap", type=int, default=None,
        help="Overlap tokens between chunks",
    )

    # --- retrieve (Phase 2) ---
    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Query the vector store"
    )
    retrieve_parser.add_argument(
        "query", type=str, help="Query text"
    )
    retrieve_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results"
    )

    args = parser.parse_args()

    chunk_size = getattr(args, "chunk_size", None) or config.chunk_size
    overlap = getattr(args, "overlap", None) or config.overlap

    if args.command == "ingest":
        doc = load(args.path)
        print(
            f"Loaded: {doc.source} "
            f"({len(doc.content)} chars, type: {doc.file_type})"
        )
        chunks = chunk(doc, chunk_size=chunk_size, overlap=overlap)
        print(f"Generated {len(chunks)} chunks")
        for c in chunks:
            print(
                f"  [{c.chunk_index}] {c.token_count} tokens: "
                f"{c.text[:60]}..."
            )

    elif args.command == "embed":
        from rag_pipeline.vectorstore import embed_chunks, PgVectorStore

        doc = load(args.path)
        print(
            f"Loaded: {doc.source} "
            f"({len(doc.content)} chars, type: {doc.file_type})"
        )

        chunks_list = chunk(doc, chunk_size=chunk_size, overlap=overlap)
        print(f"Generated {len(chunks_list)} chunks")

        print(f"Embedding with {config.embedding_model}...")
        embedded = embed_chunks(chunks_list, config)
        print(f"Embedded {len(embedded)} chunks ({config.embedding_dim} dims)")

        store = PgVectorStore(config)
        store.initialize()
        count = store.insert(embedded)
        store.close()
        print(f"Stored {count} chunks from {doc.source}")

    elif args.command == "retrieve":
        from rag_pipeline.vectorstore import embed_text, PgVectorStore

        print(f"Embedding query with {config.embedding_model}...")
        query_vec = embed_text(args.query, config)

        store = PgVectorStore(config)
        results = store.retrieve(query_vec, top_k=args.top_k)
        store.close()

        print(f"Top {len(results)} results:")
        for r in results:
            print(
                f"  [{r['similarity']:.3f}] ({r['source']}) "
                f"{r['text'][:100]}..."
            )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
