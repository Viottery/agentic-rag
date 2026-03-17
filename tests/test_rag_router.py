from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.rag_router_utils import fallback_rag_route


def test_fallback_rag_route_matches_top_level_group_from_summary() -> None:
    summary = (
        "collection=agentic_rag_docs points=100 sampled_documents=12\n"
        "- source=prts_wiki docs=12 "
        "groups=干员 (8 docs; scopes: 干员/近卫, 干员/术师); 敌人 (4 docs; scopes: 敌人/BOSS)"
    )
    task = {
        "question": "明日方舟干员银灰的定位是什么",
        "rewritten_query": "干员 银灰 定位",
        "sub_queries": [],
    }

    route = fallback_rag_route(task, summary)

    assert route.top_level_group == "干员"
    assert route.hierarchy_scope == "干员/近卫"


def test_fallback_rag_route_matches_nested_scope_when_query_mentions_it() -> None:
    summary = (
        "collection=agentic_rag_docs points=100 sampled_documents=12\n"
        "- source=prts_wiki docs=12 "
        "groups=干员 (8 docs; scopes: 干员/近卫, 干员/术师); 敌人 (4 docs; scopes: 敌人/BOSS)"
    )
    task = {
        "question": "明日方舟敌人BOSS不死的黑蛇怎么打",
        "rewritten_query": "敌人/BOSS 不死的黑蛇",
        "sub_queries": [],
    }

    route = fallback_rag_route(task, summary)

    assert route.top_level_group == "敌人"
    assert route.hierarchy_scope == "敌人/BOSS"
