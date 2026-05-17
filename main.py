"""CLI entry point for the RAG pipeline."""

import argparse
import sys

from rag_pipeline.ingestion import chunk, load


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-first, fully local RAG pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest and chunk a document")
    ingest_parser.add_argument("path", type=str, help="File path or URL to ingest")
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=512, help="Target token count per chunk"
    )
    ingest_parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap tokens between chunks"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        doc = load(args.path)
        print(f"Loaded: {doc.source} ({len(doc.content)} chars, type: {doc.file_type})")
        chunks = chunk(doc, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Generated {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c.chunk_index}] {c.token_count} tokens: {c.text[:60]}...")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
