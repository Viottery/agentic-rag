from __future__ import annotations

import argparse
import json

from app.rag.indexing import index_directory, index_text_file
from app.rag.qdrant_store import DEFAULT_COLLECTION_NAME, QdrantStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG knowledge base utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_dir_parser = subparsers.add_parser("index-dir", help="Index a directory of text files")
    index_dir_parser.add_argument("directory", help="Directory containing source files")
    index_dir_parser.add_argument("--pattern", default="**/*.txt", help="Glob pattern for files")
    index_dir_parser.add_argument("--source-name", default="local_files", help="Logical source name")
    index_dir_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    index_dir_parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap")
    index_dir_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")

    index_file_parser = subparsers.add_parser("index-file", help="Index a single text file")
    index_file_parser.add_argument("path", help="Path to a text file")
    index_file_parser.add_argument("--source-name", default="local_files", help="Logical source name")
    index_file_parser.add_argument("--document-id", default=None, help="Optional stable document id")
    index_file_parser.add_argument("--title", default=None, help="Optional display title")
    index_file_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    index_file_parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap")
    index_file_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")

    check_parser = subparsers.add_parser("check", help="Check Qdrant collection status")
    check_parser.add_argument("--collection-name", default=None, help="Optional collection override")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index-dir":
        result = index_directory(
            args.directory,
            pattern=args.pattern,
            source_name=args.source_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    if args.command == "index-file":
        chunks = index_text_file(
            args.path,
            source_name=args.source_name,
            document_id=args.document_id,
            title=args.title,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )
        print(json.dumps({"chunks": len(chunks)}, ensure_ascii=False))
        return

    if args.command == "check":
        store = QdrantStore(collection_name=args.collection_name or DEFAULT_COLLECTION_NAME)
        result = {
            "collection_name": store.collection_name,
            "collection_exists": store.collection_exists(),
            "points_count": store.count_points(),
        }
        print(json.dumps(result, ensure_ascii=False))
        return

    parser.error("unknown command")


if __name__ == "__main__":
    main()
