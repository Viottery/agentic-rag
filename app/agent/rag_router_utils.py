from __future__ import annotations

import re

from app.agent.schemas import RAGRoutePlan
from app.agent.state import SubTask


def extract_available_top_level_groups(kb_structure_summary: str) -> list[str]:
    groups: list[str] = []
    for line in kb_structure_summary.splitlines():
        if "groups=" not in line:
            continue
        group_part = line.split("groups=", 1)[1]
        for chunk in group_part.split(";"):
            cleaned = chunk.strip()
            if not cleaned or cleaned == "None":
                continue
            group_name = cleaned.split("(", 1)[0].strip()
            if group_name and group_name not in groups:
                groups.append(group_name)
    return groups


def extract_available_hierarchy_scopes(kb_structure_summary: str) -> list[str]:
    scopes: list[str] = []
    for match in re.finditer(r"scopes:\s*([^)]+)\)", kb_structure_summary):
        for item in match.group(1).split(","):
            scope = item.strip()
            if scope and scope not in scopes:
                scopes.append(scope)
    return scopes


def fallback_rag_route(task: SubTask, kb_structure_summary: str) -> RAGRoutePlan:
    query_text = " ".join(
        [
            task.get("question", ""),
            task.get("rewritten_query", ""),
            *task.get("sub_queries", []),
        ]
    )
    available_groups = extract_available_top_level_groups(kb_structure_summary)
    available_scopes = extract_available_hierarchy_scopes(kb_structure_summary)

    selected_group = ""
    for group_name in available_groups:
        if group_name and group_name in query_text:
            selected_group = group_name
            break

    selected_scope = ""
    for scope in available_scopes:
        if scope and scope in query_text:
            selected_scope = scope
            break

    if not selected_scope and selected_group:
        for scope in available_scopes:
            if scope == selected_group or scope.startswith(f"{selected_group}/"):
                selected_scope = scope
                break

    rationale = "fallback route keeps retrieval broad."
    if selected_scope:
        rationale = f"fallback matched hierarchy scope `{selected_scope}`."
    elif selected_group:
        rationale = f"fallback matched top-level group `{selected_group}`."

    return RAGRoutePlan(
        source_name="",
        top_level_group=selected_group,
        hierarchy_scope=selected_scope,
        rationale=rationale,
    )
