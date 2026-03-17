from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.mediawiki import (
    MediaWikiPage,
    html_to_text,
    _limit_reached,
    load_mediawiki_manifest,
    normalize_category_title,
    render_page_document,
    sanitize_page_title,
    sanitize_path_segment,
    sha256_text,
    strip_namespace_prefix,
)


def test_html_to_text_preserves_basic_structure() -> None:
    html = """
    <div>
      <h1>标题</h1>
      <p>第一段。</p>
      <ul><li>条目一</li><li>条目二</li></ul>
      <script>ignored()</script>
    </div>
    """

    text = html_to_text(html)

    assert "标题" in text
    assert "第一段。" in text
    assert "条目一" in text
    assert "条目二" in text
    assert "ignored" not in text


def test_sanitize_page_title_replaces_file_unsafe_chars() -> None:
    assert sanitize_page_title('A/B:C*D?"E<F>G|H') == "A_B_C_D__E_F_G_H"


def test_render_page_document_contains_metadata_header() -> None:
    page = MediaWikiPage(
        page_id=123,
        title="测试页面",
        namespace=0,
        url="https://example.org/w/%E6%B5%8B%E8%AF%95%E9%A1%B5%E9%9D%A2",
        categories=["分类A", "分类B"],
        text="这是正文。",
    )

    document = render_page_document(page, source_name="prts_wiki", knowledge_path=["明日方舟", "干员"])

    assert "Title: 测试页面" in document
    assert "Source: prts_wiki" in document
    assert "Knowledge Path: 明日方舟 / 干员" in document
    assert "Categories: 分类A, 分类B" in document
    assert "这是正文。" in document


def test_category_helpers_support_tree_storage() -> None:
    assert normalize_category_title("干员") == "分类:干员"
    assert normalize_category_title("Category:Operators", category_namespace_name="Category") == "Category:Operators"
    assert strip_namespace_prefix("分类:干员") == "干员"
    assert sanitize_path_segment("分类:干员/近卫") == "干员_近卫"


def test_limit_helper_supports_unbounded_mode() -> None:
    assert _limit_reached(0, None) is False
    assert _limit_reached(10, None) is False
    assert _limit_reached(9, 10) is False
    assert _limit_reached(10, 10) is True


def test_render_page_document_is_stable_for_incremental_hashing(tmp_path: Path) -> None:
    page = MediaWikiPage(
        page_id=123,
        title="测试页面",
        namespace=0,
        url="https://example.org/w/%E6%B5%8B%E8%AF%95%E9%A1%B5%E9%9D%A2",
        categories=["分类A"],
        text="这是正文。",
    )

    first = render_page_document(page, source_name="prts_wiki", knowledge_path=["干员"])
    second = render_page_document(page, source_name="prts_wiki", knowledge_path=["干员"])

    assert first == second
    assert sha256_text(first) == sha256_text(second)

    manifest_path = tmp_path / "_manifest.json"
    manifest_path.write_text('{"pages": []}', encoding="utf-8")
    assert load_mediawiki_manifest(tmp_path) == {"pages": []}
