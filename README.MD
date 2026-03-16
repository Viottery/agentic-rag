# Agentic-RAG

Agentic-RAG is a learning-oriented backend project for exploring how a simple LLM app can evolve into a planner-driven multi-agent workflow with retrieval, tool delegation, answer synthesis, and guarded fallback behavior.

## Project Goal

The project currently focuses on four things:

- FastAPI as the API surface
- LangGraph as the workflow orchestrator
- Qdrant-backed local RAG as the first real sub-agent capability
- A supervisor-style planner loop that can dispatch subtasks and degrade gracefully under runtime limits

The long-term direction is to replace today's mock search and mock action agents with real tools, and to grow retrieval into a richer retrieval subgraph with rewrite, routing, reranking, validation, and evidence aggregation.

## Current Workflow

The current agent is no longer a single-step router. It is a serial supervisor workflow:

```text
user input
  -> planner
  -> dispatcher
  -> rag_agent / search_agent / action_agent
  -> planner
  -> answer_generator
  -> checker
  -> END or back to planner
```

Current execution characteristics:

- `planner` decomposes the problem into `subtasks`
- `dispatcher` chooses the current task and routes by `task_type`
- `rag_agent` is real and reads from the local knowledge base
- `search_agent` is still mock, but returns a structured pseudo-search summary
- `action_agent` is still mock, but returns a structured pseudo-execution result
- `answer_generator` writes an answer draft
- `checker` decides whether the draft can be returned

More detailed flow notes live in [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md).

## Docs

- [docs/langgraph-workflow-visualization.md](/home/viottery/workspace/agentic-rag/docs/langgraph-workflow-visualization.md): current LangGraph workflow, state design, degradation strategy, and known limitations
- [docs/future-roadmap.md](/home/viottery/workspace/agentic-rag/docs/future-roadmap.md): future project direction, stage planning, and next implementation priorities

## Runtime Degradation Strategy

The workflow is intentionally bounded so it does not loop forever or burn excessive model calls.

Two runtime limits are enforced:

- `AGENT_MAX_ITERATIONS`
- `AGENT_MAX_DURATION_SECONDS`

When the planner wants to continue dispatching but one of these limits has already been reached:

1. The workflow stops creating new work.
2. It is forced into `answer_generator`.
3. The answer draft is prefixed with a clear limitation notice.
4. `checker` marks the result as a forced budget pass and returns the best-effort answer.

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

## Current Limitations

- `search_agent` is still mock and does not perform real web search.
- `action_agent` is still mock and does not perform real external actions.
- Retrieval is still baseline vector search.
- `aggregated_context` and some global fields duplicate information that also exists inside `subtasks`.
- The workflow is serial today; the dispatcher is structured for future parallel expansion, but fan-out/fan-in is not implemented yet.
- The checker currently distinguishes quality pass vs budget-forced pass, but deeper failure taxonomy is still basic.

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
- `search` mock flow
- `action` mock flow
- response shape
- completion status
- subtask type routing

## Development Notes

- If you update source documents under `data/raw`, rerun the indexing command.
- The raw documents are not indexed automatically on startup.
- Because search and action are still mocked, prompts and tests should avoid pretending they are real production integrations.
- The current system is intentionally explicit about intermediate state so planner/checker behavior can be debugged before the workflow is slimmed down.

## License

MIT
