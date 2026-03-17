from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.qdrant_store import build_structure_summary_from_payloads, render_structure_summary


def test_build_structure_summary_groups_documents_by_hierarchy() -> None:
    payloads = [
        {
            "document_id": "prts::page::1",
            "source_name": "prts_wiki",
            "top_level_group": "干员",
            "hierarchy_path": "干员/近卫",
            "title": "银灰",
        },
        {
            "document_id": "prts::page::1",
            "source_name": "prts_wiki",
            "top_level_group": "干员",
            "hierarchy_path": "干员/近卫",
            "title": "银灰",
        },
        {
            "document_id": "prts::page::2",
            "source_name": "prts_wiki",
            "top_level_group": "敌人",
            "hierarchy_path": "敌人/BOSS",
            "title": "不死的黑蛇",
        },
    ]

    summary = build_structure_summary_from_payloads(
        payloads,
        collection_name="agentic_rag_docs",
        collection_exists=True,
        point_count=12,
    )

    assert summary["collection_exists"] is True
    assert summary["point_count"] == 12
    assert summary["sampled_document_count"] == 2
    assert summary["sources"][0]["source_name"] == "prts_wiki"
    assert summary["sources"][0]["doc_count"] == 2
    assert [group["group_name"] for group in summary["sources"][0]["groups"]] == ["干员", "敌人"]
    assert summary["sources"][0]["groups"][0]["paths"] == ["干员/近卫"]


def test_render_structure_summary_outputs_compact_planner_friendly_text() -> None:
    summary = {
        "collection_name": "agentic_rag_docs",
        "collection_exists": True,
        "point_count": 24,
        "sampled_document_count": 3,
        "sources": [
            {
                "source_name": "prts_wiki",
                "doc_count": 3,
                "groups": [
                    {
                        "group_name": "干员",
                        "doc_count": 2,
                        "paths": ["干员/近卫", "干员/术师"],
                    },
                    {
                        "group_name": "敌人",
                        "doc_count": 1,
                        "paths": ["敌人/BOSS"],
                    },
                ],
            }
        ],
    }

    rendered = render_structure_summary(summary)

    assert "collection=agentic_rag_docs" in rendered
    assert "source=prts_wiki" in rendered
    assert "干员 (2 docs; scopes: 干员/近卫, 干员/术师)" in rendered
    assert "敌人 (1 docs; scopes: 敌人/BOSS)" in rendered
