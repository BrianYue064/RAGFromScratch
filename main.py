"""CLI entry point for the RAG pipeline."""

import argparse
import sys

from rag_pipeline.config import load_config
from rag_pipeline.ingestion import chunk, load, resolve_paths


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
        "paths", type=str, nargs="+",
        help="One or more file paths, URLs, or a folder to ingest"
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
        "paths", type=str, nargs="+",
        help="One or more file paths, URLs, or a folder to embed"
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

    # --- query (Phase 3) ---
    query_parser = subparsers.add_parser(
        "query", help="Answer a question using RAG"
    )
    query_parser.add_argument(
        "question", type=str, help="The question to answer"
    )
    query_parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of retrieval results (default: config.top_k)",
    )
    query_parser.add_argument(
        "--model", type=str, default=None,
        help="Override the LLM model (default: config.llm_model)",
    )
    query_parser.add_argument(
        "--provider", type=str, default=None,
        help="Override the LLM provider (ollama or lmstudio)",
    )

    args = parser.parse_args()

    chunk_size = getattr(args, "chunk_size", None) or config.chunk_size
    overlap = getattr(args, "overlap", None) or config.overlap

    if args.command == "ingest":
        resolved = resolve_paths(args.paths)
        print(f"Resolved {len(resolved)} file(s) for ingestion\n")

        for i, file_path in enumerate(resolved, 1):
            print(f"--- [{i}/{len(resolved)}] {file_path} ---")
            doc = load(str(file_path))
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
            print()

    elif args.command == "embed":
        from rag_pipeline.vectorstore import embed_chunks, PgVectorStore

        resolved = resolve_paths(args.paths)
        print(f"Resolved {len(resolved)} file(s) for embedding\n")

        store = PgVectorStore(config)
        store.initialize()
        total_stored = 0

        for i, file_path in enumerate(resolved, 1):
            print(f"--- [{i}/{len(resolved)}] {file_path} ---")
            doc = load(str(file_path))
            print(
                f"Loaded: {doc.source} "
                f"({len(doc.content)} chars, type: {doc.file_type})"
            )

            chunks_list = chunk(doc, chunk_size=chunk_size, overlap=overlap)
            print(f"Generated {len(chunks_list)} chunks")

            print(f"Embedding with {config.embedding_model}...")
            embedded = embed_chunks(chunks_list, config)
            print(f"Embedded {len(embedded)} chunks ({config.embedding_dim} dims)")

            count = store.insert(embedded)
            total_stored += count
            print(f"Stored {count} chunks from {doc.source}\n")

        store.close()
        print(f"Done — stored {total_stored} total chunks from {len(resolved)} file(s)")

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
                f"  [{r['similarity']:.3f}] ({r['source']})\n"
                f"{r['text']}\n"
            )

    elif args.command == "query":
        from rag_pipeline.generation import rag_query

        top_k = args.top_k or config.top_k
        model = args.model or config.llm_model
        provider = args.provider or config.llm_provider

        print("Querying with LLM...")
        result = rag_query(
            args.question, 
            config,
            top_k=args.top_k,
            model=args.model,
            provider=args.provider
        )
        top_sim = (
            result["sources"][0]["similarity"]
            if result["sources"] else "N/A"
        )
        print(
            f"Retrieved {len(result['sources'])} chunks "
            f"(top similarity: {top_sim})"
        )
        print()
        print("Answer:")
        print(f"  {result['answer']}")
        print()
        if result["sources"]:
            print("Sources:")
            for src in result["sources"]:
                print(
                    f"  [{src['similarity']:.3f}] ({src['source']}) "
                    f"{src['text'][:80]}..."
                )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
