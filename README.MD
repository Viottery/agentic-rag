# Agentic-RAG

Agentic-RAG is a learning-oriented backend project for building an AI agent with structured reasoning, retrieval-augmented generation, and tool execution.

The project focuses on system design rather than UI. Its goal is to explore how an LLM application can evolve from a simple prompt-based chatbot into a modular agent system with explicit workflow control.

## Project Goal

This project aims to build an Agentic RAG system based on:

- FastAPI for API delivery
- LangGraph for workflow orchestration
- ReAct-style reasoning for action planning
- Qdrant for vector retrieval
- Docker Compose for reproducible local development

The long-term direction is not only to support retrieval, but to grow the retrieval layer into a richer retrieval subgraph with better decomposition, routing, reranking, validation, and evidence aggregation.

## Current Status

The project is in an early but working stage.

Already implemented:

- FastAPI backend service
- LangGraph-based agent workflow
- Planner, retrieval, tool-executor, and responder nodes
- Basic RAG indexing pipeline
- Qdrant-backed vector search
- Structured evidence and trace output
- Docker-based local development environment

Current limitations:

- Retrieval is still a baseline vector-search implementation
- Tool execution is still mocked
- Retrieval does not yet include query rewrite, reranking, hybrid search, or evidence validation
- There is not yet an automatic startup indexing flow

## Architecture

The current workflow is:

Client
  -> FastAPI
  -> LangGraph agent
     -> planner
     -> retrieve or tool_executor
     -> responder

The retrieval path is:

Raw text files
  -> indexing
  -> embeddings
  -> Qdrant
  -> retriever
  -> agent retrieve node
  -> responder

## Project Structure

```text
agentic-rag/
├── app/
│   ├── agent/       # LangGraph workflow, nodes, prompts, state
│   ├── api/         # FastAPI routes and schemas
│   ├── core/        # configuration
│   ├── rag/         # indexing, embeddings, qdrant store, retriever, cli
│   └── tools/       # future external tools
├── data/
│   └── raw/         # source documents for indexing
├── docker-compose.yml
├── dockerfile
├── requirements.txt
└── README.MD
```

## How To Run

Start the services:

```bash
docker compose up --build
```

Build the knowledge base index from local text files:

```bash
docker compose exec app python -m app.rag.cli index-dir data/raw
```

Check whether Qdrant has indexed points:

```bash
docker compose exec app python -m app.rag.cli check
```

Send a chat request:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请根据知识库内容介绍一下这个项目的目标和整体架构。"
  }'
```

## What The Chat API Returns

The `/chat` endpoint currently returns the full agent state, including:

- `answer`
- `retrieved_docs`
- `retrieved_sources`
- `evidence`
- `intermediate_steps`
- `trace_summary`
- `error`

This is intentionally verbose in the current stage because the project is still focused on workflow validation and observability.

## Retrieval Design Direction

The current retrieval implementation is intentionally simple and maintainable:

- character-based chunking
- local embedding generation
- vector search in Qdrant
- evidence mapping back into agent state

This baseline is meant to be replaced incrementally by a more capable retrieval subgraph that may later support:

- query analysis
- sub-question decomposition
- source routing
- reranking
- evidence validation
- evidence aggregation

## Development Notes

- The raw documents under `data/raw` are not automatically indexed on startup
- If you update source files, rerun the indexing command
- Qdrant point IDs are stored as stable UUID mappings, while business-facing `chunk_id` values are preserved in payloads
- The code intentionally marks some baseline or naive implementations with comments so they can be upgraded later without changing the public module boundaries

## License

MIT
