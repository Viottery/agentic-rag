from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.indexing import (
    INDEX_MANIFEST_FILE_NAME,
    build_document_chunks,
    build_document_id_from_path,
    build_file_hierarchy_metadata,
    load_index_manifest,
    save_index_manifest,
)


def test_build_file_hierarchy_metadata_from_rooted_path() -> None:
    metadata = build_file_hierarchy_metadata(
        "data/raw/prts-wiki/干员/近卫/12345-银灰.txt",
        root_dir="data/raw/prts-wiki",
    )

    assert metadata["source_root"] == "prts-wiki"
    assert metadata["relative_path"] == "干员/近卫/12345-银灰.txt"
    assert metadata["knowledge_path"] == ["干员", "近卫"]
    assert metadata["knowledge_path_text"] == "干员 / 近卫"
    assert metadata["path_segments"] == ["干员", "近卫"]
    assert metadata["hierarchy_path"] == "干员/近卫"
    assert metadata["hierarchy_level"] == 2
    assert metadata["hierarchy_prefixes"] == ["干员", "干员/近卫"]
    assert metadata["root_group"] == "干员"
    assert metadata["leaf_group"] == "近卫"
    assert metadata["top_level_group"] == "干员"
    assert metadata["parent_path"] == "干员"


def test_build_document_id_from_relative_path_is_hierarchy_aware() -> None:
    document_id = build_document_id_from_path(
        "data/raw/prts-wiki/干员/近卫/12345-银灰.txt",
        root_dir="data/raw/prts-wiki",
    )

    assert document_id == "干员::近卫::12345-银灰"


def test_build_file_hierarchy_metadata_supports_nested_paths() -> None:
    metadata = build_file_hierarchy_metadata(
        "data/raw/prts-wiki/敌人/BOSS/炎国/99999-岁相.txt",
        root_dir="data/raw/prts-wiki",
    )

    assert metadata["knowledge_path"] == ["敌人", "BOSS", "炎国"]
    assert metadata["hierarchy_level"] == 3
    assert metadata["hierarchy_prefixes"] == ["敌人", "敌人/BOSS", "敌人/BOSS/炎国"]
    assert metadata["root_group"] == "敌人"
    assert metadata["leaf_group"] == "炎国"
    assert metadata["parent_path"] == "敌人/BOSS"


def test_index_manifest_round_trip(tmp_path: Path) -> None:
    payload = {
        "root_dir": str(tmp_path),
        "source_name": "prts_wiki",
        "documents": {
            "干员/近卫/12345-银灰.txt": {
                "document_id": "prts_wiki::page::12345",
                "content_hash": "abc",
                "chunk_count": 4,
            }
        },
    }

    manifest_path = save_index_manifest(tmp_path, payload)

    assert manifest_path.endswith(INDEX_MANIFEST_FILE_NAME)
    assert load_index_manifest(tmp_path) == payload


def test_build_document_chunks_attaches_section_metadata() -> None:
    text = """
    真理

    干员档案

    特性 [ 编辑 ]

    拥有稳定的减速与控制能力。

    技能 [ 编辑 ]

    技能二可以扩大攻击范围，并提升控制能力。
    """

    chunks = build_document_chunks(
        document_id="干员::1765-真理",
        source_name="prts_wiki",
        title="1765-真理.txt",
        text=text,
        metadata={"file_stem": "1765-真理"},
        chunk_size=48,
        chunk_overlap=0,
    )

    assert chunks

    trait_chunk = next(chunk for chunk in chunks if "减速与控制能力" in chunk.content)
    skill_chunk = next(chunk for chunk in chunks if "扩大攻击范围" in chunk.content)

    assert trait_chunk.metadata["section_title"] == "特性"
    assert trait_chunk.metadata["section_path_text"].endswith("特性")
    assert skill_chunk.metadata["section_title"] == "技能"
    assert skill_chunk.metadata["section_path"][-1] == "技能"
