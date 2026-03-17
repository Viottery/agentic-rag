from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlparse

if TYPE_CHECKING:
    import httpx


DEFAULT_MEDIAWIKI_USER_AGENT = "agentic-rag/0.1 (simple mediawiki crawler)"
MANIFEST_FILE_NAME = "_manifest.json"


@dataclass
class MediaWikiPage:
    page_id: int
    title: str
    namespace: int
    url: str
    categories: list[str]
    text: str


@dataclass
class DiscoveredMediaWikiPage:
    page_id: int
    title: str
    namespace: int
    tree_path: list[str]


class _MediaWikiHTMLToTextParser(HTMLParser):
    """
    将 MediaWiki parse API 返回的 HTML 近似转换为纯文本。

    第一版目标是得到适合索引的稳定正文文本，而不是做完美的页面还原。
    因此这里优先：
    - 保留标题、段落、列表、表格单元的换行语义
    - 跳过 script/style
    - 减少多余空白
    """

    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "br",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
    }
    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = unescape(data).strip()
        if text:
            self._parts.append(text)
            self._parts.append(" ")

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = [line.strip() for line in raw.splitlines()]
        paragraphs: list[str] = []
        current: list[str] = []

        for line in lines:
            if not line:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            current.append(line)

        if current:
            paragraphs.append(" ".join(current))

        return "\n\n".join(paragraphs).strip()


def html_to_text(html: str) -> str:
    parser = _MediaWikiHTMLToTextParser()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def sanitize_page_title(title: str) -> str:
    """
    生成适合本地文件名的页面标题。

    保留中文与常见可读字符，只替换文件系统敏感字符。
    """
    sanitized = title.strip()
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "untitled"


def strip_namespace_prefix(title: str) -> str:
    if ":" in title:
        return title.split(":", 1)[1].strip()
    return title.strip()


def sanitize_path_segment(value: str) -> str:
    return sanitize_page_title(strip_namespace_prefix(value))


def mediawiki_page_url(api_url: str, title: str) -> str:
    parsed = urlparse(api_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/w/{quote(title, safe='')}"


def _get_json(
    client: "httpx.Client",
    api_url: str,
    *,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = client.get(api_url, params=params, timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API 返回错误: {payload['error']}")
    return payload


def _limit_reached(current: int, limit: int | None) -> bool:
    return limit is not None and current >= limit


def _emit_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_pages(
    api_url: str,
    *,
    limit: int | None = None,
    namespace: int = 0,
    prefix: str | None = None,
    user_agent: str = DEFAULT_MEDIAWIKI_USER_AGENT,
) -> list[DiscoveredMediaWikiPage]:
    """
    通过 MediaWiki allpages API 发现页面。
    """
    import httpx

    if limit is not None and limit <= 0:
        return []

    pages: list[DiscoveredMediaWikiPage] = []
    continuation: str | None = None

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        while True:
            remaining = None if limit is None else max(limit - len(pages), 0)
            if remaining == 0:
                break
            batch_limit = 50 if remaining is None else min(50, remaining)
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": batch_limit,
            }
            if prefix:
                params["apprefix"] = prefix
            if continuation:
                params["apcontinue"] = continuation

            payload = _get_json(client, api_url, params=params)
            items = payload.get("query", {}).get("allpages", [])
            if not items:
                break

            for item in items:
                pages.append(
                    DiscoveredMediaWikiPage(
                        page_id=int(item["pageid"]),
                        title=str(item["title"]),
                        namespace=int(item.get("ns", namespace)),
                        tree_path=[],
                    )
                )
                if _limit_reached(len(pages), limit):
                    break
            continuation = payload.get("continue", {}).get("apcontinue")
            if not continuation or _limit_reached(len(pages), limit):
                break

    return pages if limit is None else pages[:limit]


def list_all_categories(
    api_url: str,
    *,
    limit: int | None = None,
    user_agent: str = DEFAULT_MEDIAWIKI_USER_AGENT,
) -> list[str]:
    import httpx

    if limit is not None and limit <= 0:
        return []

    categories: list[str] = []
    continuation: str | None = None

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        while True:
            remaining = None if limit is None else max(limit - len(categories), 0)
            if remaining == 0:
                break
            batch_limit = 50 if remaining is None else min(50, remaining)
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "list": "allcategories",
                "aclimit": batch_limit,
            }
            if continuation:
                params["accontinue"] = continuation

            payload = _get_json(client, api_url, params=params)
            items = payload.get("query", {}).get("allcategories", [])
            if not items:
                break

            for item in items:
                title = str(item.get("*", "")).strip()
                if title:
                    categories.append(title)
                if _limit_reached(len(categories), limit):
                    break

            continuation = payload.get("continue", {}).get("accontinue")
            if not continuation or _limit_reached(len(categories), limit):
                break

    return categories if limit is None else categories[:limit]


def normalize_category_title(title: str, *, category_namespace_name: str = "分类") -> str:
    stripped = title.strip()
    if ":" in stripped:
        return stripped
    return f"{category_namespace_name}:{stripped}"


def list_category_members(
    client: "httpx.Client",
    api_url: str,
    *,
    category_title: str,
    member_type: str,
) -> list[dict[str, Any]]:
    continuation: str | None = None
    members: list[dict[str, Any]] = []

    while True:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": member_type,
            "cmlimit": 50,
        }
        if continuation:
            params["cmcontinue"] = continuation

        payload = _get_json(client, api_url, params=params)
        items = payload.get("query", {}).get("categorymembers", [])
        members.extend(items)

        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            break

    return members


def crawl_category_tree(
    api_url: str,
    *,
    root_category: str,
    limit: int | None,
    max_depth: int,
    user_agent: str = DEFAULT_MEDIAWIKI_USER_AGENT,
    category_namespace_name: str = "分类",
    progress: bool = False,
    seen_page_ids: set[int] | None = None,
    seen_categories: set[str] | None = None,
) -> list[DiscoveredMediaWikiPage]:
    """
    从根分类递归发现页面，并记录分类路径。
    """
    import httpx

    normalized_root = normalize_category_title(root_category, category_namespace_name=category_namespace_name)
    root_segment = strip_namespace_prefix(normalized_root)
    discovered: list[DiscoveredMediaWikiPage] = []
    shared_seen_page_ids = seen_page_ids if seen_page_ids is not None else set()
    shared_seen_categories = seen_categories if seen_categories is not None else set()

    def _walk(client: "httpx.Client", category_title: str, path_segments: list[str], depth: int) -> None:
        if _limit_reached(len(discovered), limit) or depth > max_depth or category_title in shared_seen_categories:
            return
        shared_seen_categories.add(category_title)
        _emit_progress(progress, f"[crawl-mediawiki] category depth={depth} title={category_title}")

        pages = list_category_members(
            client,
            api_url,
            category_title=category_title,
            member_type="page",
        )
        for item in pages:
            page_id = int(item["pageid"])
            if page_id in shared_seen_page_ids:
                continue
            shared_seen_page_ids.add(page_id)
            discovered.append(
                DiscoveredMediaWikiPage(
                    page_id=page_id,
                    title=str(item["title"]),
                    namespace=int(item.get("ns", 0)),
                    tree_path=list(path_segments),
                )
            )
            _emit_progress(
                progress,
                f"[crawl-mediawiki] discovered page_id={page_id} title={item['title']}",
            )
            if _limit_reached(len(discovered), limit):
                return

        if depth >= max_depth:
            return

        subcategories = list_category_members(
            client,
            api_url,
            category_title=category_title,
            member_type="subcat",
        )
        for item in subcategories:
            subcategory_title = str(item["title"])
            subcategory_segment = strip_namespace_prefix(subcategory_title)
            _walk(client, subcategory_title, [*path_segments, subcategory_segment], depth + 1)
            if _limit_reached(len(discovered), limit):
                return

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        _walk(client, normalized_root, [root_segment], 0)

    return discovered if limit is None else discovered[:limit]


def crawl_all_categories_tree(
    api_url: str,
    *,
    limit: int | None,
    max_depth: int,
    user_agent: str = DEFAULT_MEDIAWIKI_USER_AGENT,
    category_namespace_name: str = "分类",
    virtual_root: str = "页面分类",
    progress: bool = False,
) -> list[DiscoveredMediaWikiPage]:
    """
    以 allcategories API 作为入口，近似映射“特殊:页面分类”的目录视图。

    说明：
    - MediaWiki 没有直接提供“整站分类树根节点”
    - 这里通过 allcategories 枚举分类，再递归其子分类和页面
    - 页面若属于多个分类，只保留首次发现的路径
    """
    all_categories = list_all_categories(
        api_url,
        limit=None,
        user_agent=user_agent,
    )
    if not all_categories:
        return []

    discovered: list[DiscoveredMediaWikiPage] = []
    seen_page_ids: set[int] = set()
    seen_categories: set[str] = set()

    def _walk(client: "httpx.Client", category_title: str, path_segments: list[str], depth: int) -> None:
        if _limit_reached(len(discovered), limit) or depth > max_depth or category_title in seen_categories:
            return
        seen_categories.add(category_title)

        pages = list_category_members(
            client,
            api_url,
            category_title=category_title,
            member_type="page",
        )
        for item in pages:
            page_id = int(item["pageid"])
            if page_id in seen_page_ids:
                continue
            seen_page_ids.add(page_id)
            discovered.append(
                DiscoveredMediaWikiPage(
                    page_id=page_id,
                    title=str(item["title"]),
                    namespace=int(item.get("ns", 0)),
                    tree_path=list(path_segments),
                )
            )
            if _limit_reached(len(discovered), limit):
                return

        if depth >= max_depth:
            return

        subcategories = list_category_members(
            client,
            api_url,
            category_title=category_title,
            member_type="subcat",
        )
        for item in subcategories:
            subcategory_title = str(item["title"])
            subcategory_segment = strip_namespace_prefix(subcategory_title)
            _walk(client, subcategory_title, [*path_segments, subcategory_segment], depth + 1)
            if _limit_reached(len(discovered), limit):
                return

    import httpx

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        for category_name in sorted(all_categories):
            category_title = normalize_category_title(
                category_name,
                category_namespace_name=category_namespace_name,
            )
            _emit_progress(progress, f"[crawl-mediawiki] root category={category_title}")
            _walk(client, category_title, [virtual_root, category_name], 0)
            if _limit_reached(len(discovered), limit):
                break

    return discovered if limit is None else discovered[:limit]


def fetch_page(
    api_url: str,
    *,
    title: str,
    page_id: int,
    namespace: int,
    user_agent: str = DEFAULT_MEDIAWIKI_USER_AGENT,
) -> MediaWikiPage:
    """
    抓取单个 MediaWiki 页面，并将 parse API 返回的 HTML 转成纯文本。
    """
    import httpx

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        payload = _get_json(
            client,
            api_url,
            params={
                "action": "parse",
                "format": "json",
                "formatversion": 2,
                "page": title,
                "prop": "text|categories",
            },
        )

    parsed = payload.get("parse", {})
    html = str(parsed.get("text", ""))
    categories = [str(item.get("category", "")) for item in parsed.get("categories", []) if item.get("category")]

    return MediaWikiPage(
        page_id=page_id,
        title=title,
        namespace=namespace,
        url=mediawiki_page_url(api_url, title),
        categories=categories,
        text=html_to_text(html),
    )


def render_page_document(
    page: MediaWikiPage,
    *,
    source_name: str,
    knowledge_path: list[str] | None = None,
) -> str:
    categories = ", ".join(page.categories) if page.categories else ""
    header_lines = [
        f"Title: {page.title}",
        f"Source: {source_name}",
        f"URL: {page.url}",
        f"Page ID: {page.page_id}",
        f"Namespace: {page.namespace}",
    ]
    if knowledge_path:
        header_lines.append(f"Knowledge Path: {' / '.join(knowledge_path)}")
    if categories:
        header_lines.append(f"Categories: {categories}")

    header = "\n".join(header_lines)
    body = page.text.strip()
    return f"{header}\n\n---\n\n{body}\n"


def load_mediawiki_manifest(output_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(output_dir) / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _existing_pages_by_id(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = manifest.get("pages", [])
    return {
        int(item["page_id"]): item
        for item in pages
        if "page_id" in item
    }


def crawl_mediawiki(
    api_url: str,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    namespace: int = 0,
    prefix: str | None = None,
    root_categories: list[str] | None = None,
    all_categories_tree: bool = False,
    max_depth: int = 2,
    source_name: str = "mediawiki",
    user_agent: str = DEFAULT_MEDIAWIKI_USER_AGENT,
    category_namespace_name: str = "分类",
    progress: bool = True,
) -> dict[str, Any]:
    """
    抓取 MediaWiki 页面并落盘为本地 txt 文档。

    当前第一版流程刻意保持简单：
    - 发现页面
    - 拉取页面 HTML
    - 近似转换为纯文本
    - 落盘为 txt，供现有 index-dir 复用
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = load_mediawiki_manifest(target_dir)
    previous_pages_by_id = _existing_pages_by_id(previous_manifest)

    normalized_root_categories = [item for item in (root_categories or []) if item.strip()]

    if normalized_root_categories and all_categories_tree:
        raise ValueError("root_categories 与 all_categories_tree 不能同时启用")

    if normalized_root_categories:
        discovered_pages: list[DiscoveredMediaWikiPage] = []
        seen_page_ids: set[int] = set()
        seen_categories: set[str] = set()
        remaining_limit = limit

        for root_category in normalized_root_categories:
            _emit_progress(progress, f"[crawl-mediawiki] start root category={root_category}")
            category_pages = crawl_category_tree(
                api_url,
                root_category=root_category,
                limit=remaining_limit,
                max_depth=max_depth,
                user_agent=user_agent,
                category_namespace_name=category_namespace_name,
                progress=progress,
                seen_page_ids=seen_page_ids,
                seen_categories=seen_categories,
            )
            discovered_pages.extend(category_pages)
            if limit is not None:
                remaining_limit = max(limit - len(discovered_pages), 0)
                if remaining_limit == 0:
                    break
    elif all_categories_tree:
        discovered_pages = crawl_all_categories_tree(
            api_url,
            limit=limit,
            max_depth=max_depth,
            user_agent=user_agent,
            category_namespace_name=category_namespace_name,
            progress=progress,
        )
    else:
        _emit_progress(progress, "[crawl-mediawiki] start allpages discovery")
        discovered_pages = list_pages(
            api_url,
            limit=limit,
            namespace=namespace,
            prefix=prefix,
            user_agent=user_agent,
        )

    written_files = 0
    skipped_files = 0
    deleted_files = 0
    manifest: list[dict[str, Any]] = []
    _emit_progress(progress, f"[crawl-mediawiki] discovered total pages={len(discovered_pages)}")
    current_page_ids: set[int] = set()

    for item in discovered_pages:
        current_page_ids.add(item.page_id)
        page = fetch_page(
            api_url,
            title=item.title,
            page_id=item.page_id,
            namespace=item.namespace,
            user_agent=user_agent,
        )
        if not page.text.strip():
            continue

        filename = f"{page.page_id}-{sanitize_page_title(page.title)}.txt"
        relative_dir = Path(*[sanitize_path_segment(segment) for segment in item.tree_path]) if item.tree_path else Path()
        file_dir = target_dir / relative_dir
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / filename
        relative_path = file_path.relative_to(target_dir).as_posix()
        document_id = f"{source_name}::page::{page.page_id}"
        rendered_document = render_page_document(
            page,
            source_name=source_name,
            knowledge_path=item.tree_path,
        )
        content_hash = sha256_text(rendered_document)
        previous_page = previous_pages_by_id.get(page.page_id)

        if (
            previous_page
            and previous_page.get("content_hash") == content_hash
            and previous_page.get("relative_path") == relative_path
            and file_path.exists()
        ):
            skipped_files += 1
            _emit_progress(
                progress,
                f"[crawl-mediawiki] skip unchanged page_id={page.page_id} title={page.title}",
            )
        else:
            previous_relative_path = previous_page.get("relative_path") if previous_page else None
            if previous_relative_path and previous_relative_path != relative_path:
                old_file_path = target_dir / previous_relative_path
                if old_file_path.exists():
                    old_file_path.unlink()
                    deleted_files += 1
                    _emit_progress(
                        progress,
                        f"[crawl-mediawiki] moved old_file={old_file_path}",
                    )

            file_path.write_text(
                rendered_document,
                encoding="utf-8",
            )
            written_files += 1
            _emit_progress(
                progress,
                f"[crawl-mediawiki] wrote {written_files} file={file_path}",
            )
        manifest.append(
            {
                "page_id": page.page_id,
                "title": page.title,
                "namespace": page.namespace,
                "url": page.url,
                "file_path": str(file_path),
                "relative_path": relative_path,
                "document_id": document_id,
                "tree_path": item.tree_path,
                "categories": page.categories,
                "content_hash": content_hash,
            }
        )

    for page_id, previous_page in previous_pages_by_id.items():
        if page_id in current_page_ids:
            continue
        previous_relative_path = previous_page.get("relative_path")
        if not previous_relative_path:
            continue
        old_file_path = target_dir / previous_relative_path
        if old_file_path.exists():
            old_file_path.unlink()
            deleted_files += 1
            _emit_progress(
                progress,
                f"[crawl-mediawiki] deleted stale file={old_file_path}",
            )

    manifest_path = target_dir / MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "api_url": api_url,
                "source_name": source_name,
                "namespace": namespace,
                "prefix": prefix,
                "root_categories": normalized_root_categories,
                "all_categories_tree": all_categories_tree,
                "max_depth": max_depth,
                "limit": limit,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "pages": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "discovered_pages": len(discovered_pages),
        "written_files": written_files,
        "skipped_files": skipped_files,
        "deleted_files": deleted_files,
        "output_dir": str(target_dir),
        "manifest_path": str(manifest_path),
    }
