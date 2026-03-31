# Agentic-RAG

Agentic-RAG is a learning-oriented backend project for exploring how a simple LLM app can evolve into a planner-driven agent system with retrieval, tool delegation, answer synthesis, and guarded fallback behavior.

## Project Goal

The project currently focuses on four things:

- FastAPI as the API surface
- LangGraph as the workflow orchestrator
- Qdrant-backed local RAG and Tavily-backed online search as the first real information abilities
- A supervisor-style planning loop that can decompose work, route to the right capability, and degrade gracefully under runtime limits

The next practical goal is not to hardcode more and more fixed workflows, but to make retrieval and tool use more structured so the agent can plan its own path through a stable capability graph.

In the near term, this means:

- making local RAG hierarchical, route-aware, and eventually multi-recall + rerank
- treating online search as a fallback or explicit search path with a more mature retrieval pipeline
- using LangGraph fan-out / fan-in to parallelize complex retrieval and search tasks
- keeping grounded answers, citations, and verification as hard constraints

In the longer term, this means evolving the system from an "agentic RAG demo" into a more complete agent runtime that can search, read and edit files, run code, call tools, build knowledge bases, and iteratively solve complex tasks.

## Current Workflow

The current agent is no longer a single-step router. It is a serial supervisor workflow that provides a capability skeleton rather than a hardcoded business workflow:

```text
user input
  -> planner
  -> dispatcher
  -> query_refiner
  -> rag_agent / search_agent / action_agent
  -> planner
  -> answer_generator
  -> citation_mapper
  -> verifier
  -> checker
  -> END or back to planner
```

Current execution characteristics:

- `planner` decomposes the problem into `subtasks`
- `dispatcher` chooses the current task and routes by `task_type`
- `query_refiner` rewrites and optionally decomposes search / retrieval queries while trying to preserve the user's language and avoid over-splitting
- `rag_agent` is real and reads from the local knowledge base
- `search_agent` can use Tavily when configured, otherwise falls back to mock search; for real search it now does a lightweight web-RAG pass: search, extract, chunk, rerank
- `action_agent` is still mock, but remains as the future execution boundary
- `answer_generator` writes an answer draft from compacted task context
- `citation_mapper` maps answer paragraphs to evidence and attaches source labels, preferring task-local evidence before global fallback
- `verifier` performs lightweight citation coverage and unsupported-section checks
- `checker` decides whether the draft can be returned

Important design note:

- the graph defines reusable node roles and safety boundaries
- it does not try to enumerate every future task workflow in advance
- except for explicitly engineered pipelines such as knowledge-base ingestion, future complex task workflows are expected to be planned by the agent itself on top of this capability graph

More detailed flow notes live in [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md).

## Docs

- [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md): current LangGraph workflow, state design, degradation strategy, and known limitations
- [docs/future-roadmap.md](/home/viottery/workspace/agentic-rag/docs/future-roadmap.md): near-term RAG implementation direction and long-term agent system design targets

## Runtime Degradation Strategy

The workflow is intentionally bounded so it does not loop forever or burn excessive model calls.

Two runtime limits are enforced:

- `AGENT_MAX_ITERATIONS`
- `AGENT_MAX_DURATION_SECONDS`

When the planner wants to continue dispatching but one of these limits has already been reached:

1. The workflow stops creating new work.
2. It is forced into `answer_generator`.
3. The answer draft is prefixed with a clear limitation notice.
4. `citation_mapper` and `verifier` still run so the result keeps grounding metadata.
5. `checker` marks the result as a forced budget pass and returns the best-effort answer.

This gives the system a predictable fallback:

- normal case: `quality_pass`
- degraded case: `forced_budget_pass`
- insufficient case before fallback: `quality_fail`

This is a deliberate tradeoff. It favors bounded execution and observability over perfect completeness.

## State Design

The `/chat` endpoint currently returns the full `AgentState` because the project is still in workflow-validation mode.

Important top-level fields include:

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

Important subtask fields include:

- `task_id`
- `task_type`
- `status`
- `result`
- `evidence`
- `error`
- `degraded`
- `degraded_reason`

Current design tradeoff:

- It is more verbose than a production state model.
- That is intentional for now because it makes planning loops, mock boundaries, and fallback behavior easier to inspect.
- Later, some global derived fields can be slimmed down once behavior is stable.

## Code Conventions

Current repository conventions for this phase are:

- Keep the workflow explicit and debuggable rather than prematurely minimal.
- Prefer small helper functions in workflow nodes when the logic is reused or safety-critical.
- Use short docstrings and comments to explain orchestration intent, fallback behavior, or non-obvious trust constraints.
- Treat search results, retrieved snippets, and generated drafts as untrusted data inside prompts.
- Prefer conservative grounding: it is acceptable for citation coverage to be stricter if that prevents weak evidence from being attached to the answer.

## Current Limitations

- `search_agent` can perform real Tavily search when configured, but deeper result filtering, extraction, and validation are still basic.
- `action_agent` is still mock and does not perform real external actions.
- Local retrieval is still baseline vector search rather than hierarchical routing with multi-recall and reranking over structured knowledge domains.
- Citation mapping and verification are currently lightweight heuristic checks, not full claim-level factual verification.
- Query rewriting is intentionally conservative now, but it is still heuristic rather than retrieval-policy-aware.
- `aggregated_context` and some global fields duplicate information that also exists inside `subtasks`.
- The workflow is serial today; the dispatcher is structured for future parallel expansion, but retrieval/search-level fan-out/fan-in is not implemented yet.
- The checker currently distinguishes quality pass vs budget-forced pass, but deeper failure taxonomy is still basic.
- Tool use is still at an early stage; the current graph has an execution boundary, but not yet a general-purpose agent runtime comparable to coding agents or desktop agent assistants.

## Project Structure

```text
agentic-rag/
├── app/
│   ├── agent/       # LangGraph workflow, nodes, prompts, state, schemas
│   ├── api/         # FastAPI routes and schemas
│   ├── core/        # configuration
│   ├── rag/         # indexing, embeddings, qdrant store, retriever, cli
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
- subtask type routing

## Development Notes

- If you update source documents under `data/raw`, rerun the indexing command.
- The raw documents are not indexed automatically on startup.
- `crawl-mediawiki` is the current minimal knowledge-ingestion demo path for wiki-style sources such as PRTS Wiki.
- `crawl-mediawiki --root-category ...` enables a simple category-tree crawl so local files can mirror wiki knowledge structure.
- `search_agent` already supports real Tavily search, but tests should still treat search quality as heuristic rather than fully production-hardened.
- `action_agent` is still mocked, so prompts and tests should avoid pretending it is a real external executor.
- The current system is intentionally explicit about intermediate state so planner/checker behavior can be debugged before the workflow is slimmed down.

## License

MIT
