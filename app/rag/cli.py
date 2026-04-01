from __future__ import annotations

import argparse
import json

from app.rag.embeddings import describe_active_embedding_runtime, describe_embedding_runtime
from app.rag.indexing import index_directory, index_text_file
from app.rag.mediawiki import crawl_mediawiki
from app.rag.qdrant_store import DEFAULT_COLLECTION_NAME, QdrantStore


def _add_embedding_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model name for this run",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "torch", "openvino"],
        default=None,
        help="Override embedding backend for this run",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "gpu", "cpu", "cuda", "xpu"],
        default=None,
        help="Override embedding device for this run",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Shortcut for --embedding-device gpu",
    )


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
    index_dir_parser.add_argument("--quiet", action="store_true", help="Disable indexing progress output")
    _add_embedding_runtime_args(index_dir_parser)

    index_file_parser = subparsers.add_parser("index-file", help="Index a single text file")
    index_file_parser.add_argument("path", help="Path to a text file")
    index_file_parser.add_argument("--source-name", default="local_files", help="Logical source name")
    index_file_parser.add_argument("--document-id", default=None, help="Optional stable document id")
    index_file_parser.add_argument("--title", default=None, help="Optional display title")
    index_file_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    index_file_parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap")
    index_file_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    index_file_parser.add_argument("--quiet", action="store_true", help="Disable indexing progress output")
    _add_embedding_runtime_args(index_file_parser)

    crawl_mediawiki_parser = subparsers.add_parser("crawl-mediawiki", help="Crawl a MediaWiki site into local txt files")
    crawl_mediawiki_parser.add_argument("api_url", help="MediaWiki api.php endpoint")
    crawl_mediawiki_parser.add_argument("output_dir", help="Directory to write crawled txt files")
    crawl_mediawiki_parser.add_argument("--limit", type=int, default=None, help="Optional max pages to crawl; omit for no hard limit")
    crawl_mediawiki_parser.add_argument("--namespace", type=int, default=0, help="MediaWiki namespace")
    crawl_mediawiki_parser.add_argument("--prefix", default=None, help="Optional title prefix filter")
    crawl_mediawiki_parser.add_argument(
        "--root-category",
        action="append",
        default=[],
        help="Root category for tree crawl. Repeat this flag to crawl multiple top-level directories.",
    )
    crawl_mediawiki_parser.add_argument(
        "--all-categories-tree",
        action="store_true",
        help="Build a tree crawl from MediaWiki allcategories, approximating 特殊:页面分类",
    )
    crawl_mediawiki_parser.add_argument("--max-depth", type=int, default=2, help="Max category tree depth when using --root-category")
    crawl_mediawiki_parser.add_argument(
        "--category-namespace-name",
        default="分类",
        help="Localized category namespace name, used when --root-category omits namespace prefix",
    )
    crawl_mediawiki_parser.add_argument("--source-name", default="mediawiki", help="Logical source name")
    crawl_mediawiki_parser.add_argument("--user-agent", default=None, help="Optional crawler user agent")
    crawl_mediawiki_parser.add_argument("--quiet", action="store_true", help="Disable progress output")
    crawl_mediawiki_parser.add_argument("--index", action="store_true", help="Index crawled txt files after crawling")
    crawl_mediawiki_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for optional indexing")
    crawl_mediawiki_parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for optional indexing")
    crawl_mediawiki_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size for optional indexing")
    _add_embedding_runtime_args(crawl_mediawiki_parser)

    check_parser = subparsers.add_parser("check", help="Check Qdrant collection status")
    check_parser.add_argument("--collection-name", default=None, help="Optional collection override")

    runtime_parser = subparsers.add_parser("embedding-runtime", help="Show the embedding runtime that indexing will use")
    _add_embedding_runtime_args(runtime_parser)

    return parser


def _resolve_embedding_cli_args(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    model_name = getattr(args, "embedding_model", None)
    backend = getattr(args, "embedding_backend", None)
    device = getattr(args, "embedding_device", None)

    if getattr(args, "force_gpu", False):
        device = "gpu"

    return model_name, backend, device


def _ensure_force_gpu_honored(
    args: argparse.Namespace,
    *,
    model_name: str | None,
    backend: str | None,
    device: str | None,
) -> None:
    if not getattr(args, "force_gpu", False):
        return

    runtime = describe_active_embedding_runtime(
        model_name=model_name,
        backend=backend,
        device=device,
    )
    resolved_device = str(runtime.get("device", "")).strip().lower()
    if resolved_device in {"gpu", "cuda", "xpu"}:
        return

    raise RuntimeError(
        "检测到 --force-gpu，但模型真实初始化后仍未使用 GPU。"
        f"resolved_runtime={runtime}。"
        "这通常意味着 GPU 虽然可见，但该 embedding 模型在当前 OpenVINO/Torch GPU 组合上无法成功编译或初始化。"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index-dir":
        embedding_model_name, embedding_backend, embedding_device = _resolve_embedding_cli_args(args)
        _ensure_force_gpu_honored(
            args,
            model_name=embedding_model_name,
            backend=embedding_backend,
            device=embedding_device,
        )
        result = index_directory(
            args.directory,
            pattern=args.pattern,
            source_name=args.source_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            embedding_model_name=embedding_model_name,
            embedding_backend=embedding_backend,
            embedding_device=embedding_device,
            progress=not args.quiet,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    if args.command == "index-file":
        embedding_model_name, embedding_backend, embedding_device = _resolve_embedding_cli_args(args)
        _ensure_force_gpu_honored(
            args,
            model_name=embedding_model_name,
            backend=embedding_backend,
            device=embedding_device,
        )
        chunks = index_text_file(
            args.path,
            source_name=args.source_name,
            document_id=args.document_id,
            title=args.title,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            embedding_model_name=embedding_model_name,
            embedding_backend=embedding_backend,
            embedding_device=embedding_device,
            progress=not args.quiet,
        )
        print(json.dumps({"chunks": len(chunks)}, ensure_ascii=False))
        return

    if args.command == "crawl-mediawiki":
        embedding_model_name, embedding_backend, embedding_device = _resolve_embedding_cli_args(args)
        _ensure_force_gpu_honored(
            args,
            model_name=embedding_model_name,
            backend=embedding_backend,
            device=embedding_device,
        )
        result = crawl_mediawiki(
            args.api_url,
            args.output_dir,
            limit=args.limit,
            namespace=args.namespace,
            prefix=args.prefix,
            root_categories=args.root_category,
            all_categories_tree=args.all_categories_tree,
            max_depth=args.max_depth,
            source_name=args.source_name,
            user_agent=args.user_agent or "agentic-rag/0.1 (simple mediawiki crawler)",
            category_namespace_name=args.category_namespace_name,
            progress=not args.quiet,
        )
        if args.index:
            indexing_result = index_directory(
                args.output_dir,
                source_name=args.source_name,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                embedding_model_name=embedding_model_name,
                embedding_backend=embedding_backend,
                embedding_device=embedding_device,
                progress=not args.quiet,
            )
            result["indexing"] = indexing_result
        print(json.dumps(result, ensure_ascii=False))
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

    if args.command == "embedding-runtime":
        embedding_model_name, embedding_backend, embedding_device = _resolve_embedding_cli_args(args)
        _ensure_force_gpu_honored(
            args,
            model_name=embedding_model_name,
            backend=embedding_backend,
            device=embedding_device,
        )
        result = describe_active_embedding_runtime(
            model_name=embedding_model_name,
            backend=embedding_backend,
            device=embedding_device,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    parser.error("unknown command")


if __name__ == "__main__":
    main()
