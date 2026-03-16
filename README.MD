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

- making local RAG hierarchical and route-aware
- treating online search as a fallback or explicit search path
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
- Local retrieval is still baseline vector search rather than hierarchical routing over structured knowledge domains.
- Citation mapping and verification are currently lightweight heuristic checks, not full claim-level factual verification.
- Query rewriting is intentionally conservative now, but it is still heuristic rather than retrieval-policy-aware.
- `aggregated_context` and some global fields duplicate information that also exists inside `subtasks`.
- The workflow is serial today; the dispatcher is structured for future parallel expansion, but fan-out/fan-in is not implemented yet.
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

For model download acceleration inside Docker, the project currently uses:

- `HF_ENDPOINT=https://hf-mirror.com`
- host-mounted Hugging Face caches via `docker-compose.yml`

## Run Locally

Start services:

```bash
docker compose up -d app
```

Index local files:

```bash
docker compose exec app python -m app.rag.cli index-dir data/raw
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
- completion status
- subtask type routing

## Development Notes

- If you update source documents under `data/raw`, rerun the indexing command.
- The raw documents are not indexed automatically on startup.
- `search_agent` already supports real Tavily search, but tests should still treat search quality as heuristic rather than fully production-hardened.
- `action_agent` is still mocked, so prompts and tests should avoid pretending it is a real external executor.
- The current system is intentionally explicit about intermediate state so planner/checker behavior can be debugged before the workflow is slimmed down.

## License

MIT
