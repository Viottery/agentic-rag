# Agentic-RAG

Agentic-RAG is a learning-oriented backend project for evolving a simple RAG demo into a skill-oriented agent runtime with fast-path answering, planner-driven task decomposition, local retrieval, tool execution, and grounding validation.

## Project Goal

The project currently focuses on four things:

- FastAPI as the API surface
- LangGraph as the runtime orchestrator
- Qdrant-backed local RAG and Tavily-backed online search as the first real information abilities
- A fast-path + planner-loop architecture that keeps simple questions short and complex tasks structured

The next practical goal is not to hardcode more and more fixed workflows, but to package retrieval and tools into stable skills so the planner can reason over a smaller, cleaner action space.

In the near term, this means:

- turning local RAG into a single callable skill with routing, recall, and rerank hidden behind one interface
- adding a fast path so simple questions can return directly without entering the planner loop
- restructuring the agent around `planner -> atomic subtasks -> skill execution -> planner feedback`
- upgrading validation from lightweight citation checks into grounding validation over tool results and answers

In the longer term, this means evolving the system from an "agentic RAG demo" into a more complete agent runtime that can search, read and edit files, run code, call tools, build knowledge bases, and iteratively solve complex tasks.

## Target Runtime

The system is being refactored from a serial supervisor workflow into a two-path runtime:

```text
user input
  -> fast gate
  -> direct answer

user input
  -> fast gate
  -> planner
  -> subtask decomposition
  -> skill executor(s)
  -> structured results
  -> planner
  -> validator
  -> answer synthesizer
  -> validator
  -> END or back to planner
```

Current design characteristics:

- `fast gate` decides whether the request can return directly, should use a single skill, or must enter the planner loop
- `planner` only performs global task decomposition, executor selection, and missing-information decisions
- `skill executor` handles atomic tasks with minimal local decision-making
- `local_kb_retrieve` is becoming the main local RAG skill, encapsulating routing, hybrid recall, and reranking
- `validator` is being repositioned as a grounding component that checks result-answer alignment and reduces hallucination
- simple questions should avoid the multi-step planner loop whenever possible

Important design note:

- the runtime should expose stable skills rather than many low-level node roles
- the planner should reason over a smaller set of structured capabilities
- future complex workflows should emerge from planning over skills, not from endless fixed graph branches

More detailed flow notes live in [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md).

## Docs

- [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md): target runtime architecture, fast/normal paths, planner boundary, skill runtime, and validator role
- [docs/future-roadmap.md](/home/viottery/workspace/agentic-rag/docs/future-roadmap.md): staged migration plan for fast gate, skill abstraction, local RAG packaging, reranker, and validation

## Runtime Strategy

The runtime is intentionally bounded so it does not loop forever or burn excessive model calls.

Two runtime limits are enforced:

- `AGENT_MAX_ITERATIONS`
- `AGENT_MAX_DURATION_SECONDS`

When the planner loop wants to continue but one of these limits has already been reached:

1. The runtime stops creating new work.
2. It is forced into answer synthesis.
3. The answer is prefixed with a clear limitation notice.
4. Validation still runs so the result keeps grounding metadata.
5. The system returns the best-effort answer rather than spinning.

This gives the system a predictable fallback:

- normal case: `quality_pass`
- degraded case: `forced_budget_pass`
- insufficient case before fallback: `quality_fail`

This is a deliberate tradeoff. It favors bounded execution and observability over perfect completeness.

## State Design

The `/chat` endpoint currently returns a verbose state because the project is still in architecture-transition mode.

Important top-level fields today include:

- `subtasks`
- `planner_control`
- `current_task`
- `aggregated_context`
- `evidence`
- `answer_draft`
- `grounded_answer`
- `citations`
- `verification_result`
- `checker_result`
- `trace_summary`
- `iteration_count`
- `max_iterations`
- `max_duration_seconds`
- `answer`
- `status`

The next state shape is expected to move toward:

- `fast_path_decision`
- `planner_state`
- `subtasks`
- `skill_results`
- `evidence`
- `answer`
- `validation`
- `trace`

Important subtask fields today include:

- `task_id`
- `task_type`
- `status`
- `result`
- `evidence`
- `error`
- `degraded`
- `degraded_reason`

Current design tradeoff:

- It is more verbose than the target runtime model.
- That is intentional for now because it makes planner loops, skill packaging, and validation behavior easier to inspect.
- Later, node-shaped fields can be slimmed down in favor of task/skill/result objects.

## Code Conventions

Current repository conventions for this phase are:

- Keep the runtime explicit and debuggable rather than prematurely abstract.
- Prefer packaging reusable retrieval or tool logic as skills instead of spreading it across planner prompts.
- Use short docstrings and comments to explain orchestration intent, fallback behavior, or non-obvious trust constraints.
- Treat search results, retrieved snippets, and generated drafts as untrusted data inside prompts.
- Prefer conservative grounding: it is acceptable for validation to be stricter if that prevents weak evidence from being attached to the answer.

## Current Limitations

- The codebase still contains older node-oriented workflow structure and has not fully migrated to the new skill runtime.
- `local_kb_retrieve` is still being packaged; some RAG logic remains spread across planner-facing nodes.
- The fast path gate is not fully implemented yet.
- Validation is still lighter than the target grounding-validator design.
- Tool execution is still at an early stage and has not yet converged on the same skill contract as local retrieval.
- Parent-child indexing remains a TODO rather than an active delivery item.

## Project Structure

```text
agentic-rag/
├── app/
│   ├── agent/       # LangGraph workflow, nodes, prompts, state, schemas
│   ├── api/         # FastAPI routes and schemas
│   ├── core/        # configuration
│   ├── rag/         # indexing, embeddings, qdrant store, retriever, reranker, cli
│   └── tools/       # future external tools
├── data/
│   └── raw/         # source documents for indexing
├── docs/
│   ├── future-roadmap.md
│   └── langgraph-workflow-visualization.md
├── tests/
│   ├── e2e_chat_flow.py
│   └── run_full_stack_smoke.sh
├── docker-compose.yml
├── dockerfile
├── requirements.txt
└── README.MD
```

## Configuration

Important environment variables:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
OPENAI_TEMPERATURE=0
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
EMBEDDING_BACKEND=auto
EMBEDDING_DEVICE=auto
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_DEVICE=auto
RERANKER_MAX_CANDIDATES=12
TAVILY_API_KEY=
TAVILY_SEARCH_DEPTH=basic
TAVILY_MAX_RESULTS=5
TAVILY_EXTRACT_TOP_K=2
TAVILY_EXTRACT_DEPTH=basic
TAVILY_CHUNK_SIZE=900
TAVILY_CHUNK_OVERLAP=120
TAVILY_MAX_CHUNKS_PER_SEARCH=4

AGENT_MAX_ITERATIONS=6
AGENT_MAX_DURATION_SECONDS=90
```

Embedding runtime selection:

- `EMBEDDING_BACKEND=auto` will prefer Intel GPU via OpenVINO when available, otherwise fall back to Torch
- `EMBEDDING_DEVICE=auto` will prefer Intel GPU, then Nvidia CUDA, then CPU
- explicit override examples:
  - `EMBEDDING_BACKEND=openvino` + `EMBEDDING_DEVICE=gpu`
  - `EMBEDDING_BACKEND=torch` + `EMBEDDING_DEVICE=cuda`
  - `EMBEDDING_BACKEND=torch` + `EMBEDDING_DEVICE=cpu`
- `resolve_embedding_runtime()` reflects the preferred runtime choice, while `get_embedding_model()` is the final authority because model export / compile may still fail and trigger a fallback

For model download acceleration inside Docker, the project currently uses:

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HUB_DISABLE_XET=1` to prevent `huggingface_hub` from bypassing the mirror through the Xet/CAS download path
- host-mounted Hugging Face caches via `docker-compose.yml`

## Intel GPU Notes

For the WSL2 + Docker + Intel GPU path used in this repository, the runtime is split into two layers:

- device visibility:
  - OpenVINO can enumerate `GPU`
  - `clinfo` can enumerate `Intel(R) OpenCL Graphics`
- model execution:
  - some SentenceTransformers models may still fail during OpenVINO GPU export / compile
  - in `auto` mode, the embedding loader will log the failure reason and fall back to Torch

Recommended verification commands:

```bash
docker exec agentic-rag-app clinfo | sed -n '1,80p'
docker exec agentic-rag-app python -c "from openvino import Core; core = Core(); print(core.available_devices); print(core.get_versions('GPU'))"
docker exec agentic-rag-app python -c "from app.rag.embeddings import resolve_embedding_runtime; print(resolve_embedding_runtime())"
```

If you want to validate actual model execution rather than only device visibility, test a real encode path:

```bash
docker exec agentic-rag-app python -c "from app.rag.embeddings import embed_query; vec = embed_query('embedding runtime smoke test'); print(len(vec))"
```

Current observed behavior on this machine:

- Intel GPU device visibility is working
- OpenVINO GPU plugin loads successfully
- some embedding models can still fail at OpenVINO GPU program build time
- when that happens, the code falls back automatically instead of breaking the RAG pipeline

## Run Locally

Start services:

```bash
docker compose up -d app
```

Index local files:

```bash
docker compose exec app python -m app.rag.cli index-dir data/raw
```

Crawl a simple MediaWiki knowledge source into local raw documents:

```bash
docker compose exec app python -m app.rag.cli crawl-mediawiki \
  "https://prts.wiki/api.php" \
  data/raw/prts-wiki \
  --source-name prts_wiki
```

If the wiki has a meaningful category tree, you can also crawl it into a matching local directory hierarchy:

```bash
docker compose exec app python -m app.rag.cli crawl-mediawiki \
  "https://prts.wiki/api.php" \
  data/raw/prts-wiki \
  --root-category 干员 \
  --root-category 敌人 \
  --max-depth 3 \
  --source-name prts_wiki
```

That tree mode writes pages under category-path folders instead of a single flat directory. Repeat `--root-category` to crawl multiple top-level directories.

For sites like PRTS Wiki, you can also approximate `特殊:页面分类` through the MediaWiki category APIs:

```bash
docker compose exec app python -m app.rag.cli crawl-mediawiki \
  "https://prts.wiki/api.php" \
  data/raw/prts-wiki \
  --all-categories-tree \
  --max-depth 3 \
  --source-name prts_wiki
```

Notes:

- `--limit` is now optional; omit it to remove the hard page cap
- `--all-categories-tree` is the closest current mode to a full category-directory crawl
- `--root-category` can be repeated, which is useful for focused wiki demos such as `干员` + `敌人`
- `首页` is the main page title on PRTS, but full knowledge crawling is better driven by MediaWiki APIs than by recursively scraping homepage links

Or crawl and index in one step:

```bash
docker compose exec app python -m app.rag.cli crawl-mediawiki \
  "https://prts.wiki/api.php" \
  data/raw/prts-wiki \
  --source-name prts_wiki \
  --index
```

Check Qdrant:

```bash
docker compose exec app python -m app.rag.cli check
```

Send a sample request:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请根据知识库内容介绍一下这个项目的目标和整体架构。"
  }'
```

## Test

Run the lightweight end-to-end test cases:

```bash
python tests/e2e_chat_flow.py --base-url http://127.0.0.1:8000
```

Run the full smoke flow:

```bash
bash tests/run_full_stack_smoke.sh
```

These tests currently validate:

- `rag` flow
- `search` flow
- `action` mock flow
- response shape
- mediawiki text normalization helpers
- completion status
- subtask execution routing

## Development Notes

- If you update source documents under `data/raw`, rerun the indexing command.
- The raw documents are not indexed automatically on startup.
- `crawl-mediawiki` is the current minimal knowledge-ingestion demo path for wiki-style sources such as PRTS Wiki.
- `crawl-mediawiki --root-category ...` enables a simple category-tree crawl so local files can mirror wiki knowledge structure.
- `search_agent` already supports real Tavily search, but tests should still treat search quality as heuristic rather than fully production-hardened.
- `action_agent` is still mocked, so prompts and tests should avoid pretending it is a real external executor.
- The current system is intentionally explicit about intermediate state so planner, skill execution, and validation behavior can be debugged before the runtime is slimmed down.

## License

MIT
