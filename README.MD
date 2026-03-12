# Agentic-RAG

Agentic-RAG is a learning-oriented project focused on designing and implementing an **AI Agent system** using structured reasoning workflows and retrieval-augmented knowledge access.

The project explores how large language models can be organized into a modular agent architecture instead of being used as simple prompt-driven applications.

---

## Project Objective

The goal of this project is to study how **LLM-based agents** can be designed as structured systems with explicit reasoning workflows.

Key objectives include:

- Designing a **graph-based agent architecture**
- Implementing **ReAct-style reasoning workflows**
- Integrating **retrieval-augmented generation (RAG)**
- Supporting **external tool invocation**
- Building a **clean and extensible backend architecture**

The focus is on **system design and engineering practices**, rather than building a simple chatbot interface.

---

## Core Technologies

The system is built using the following technologies.

**Backend Framework**
- FastAPI

**Agent Framework**
- LangGraph
- LangChain

**Retrieval Infrastructure**
- Qdrant vector database

**Containerized Environment**
- Docker
- Docker Compose

**Language**
- Python 3.11

---

## Methodology

The project adopts a **graph-based agent workflow** in which reasoning and actions are represented as nodes in an execution graph.

Instead of a single LLM call, the agent executes a structured workflow that may include:

- reasoning and planning
- retrieval from a vector database
- tool execution
- response generation

This design provides explicit control flow and improves the transparency and extensibility of the agent system.

---

## System Architecture

The intended architecture is organized around a central agent workflow.

Client  
│  
▼  
FastAPI API  
│  
▼  
LangGraph Agent Workflow  
│  
├── Planner (ReAct reasoning)  
│  
├── Retrieval Node  
│     │  
│     ▼  
│   Vector Database (Qdrant)  
│  
└── Tool Execution Node  
│  
▼  
Response Generation  

This workflow allows the system to perform iterative reasoning, retrieval, and action execution before generating the final response.

---

## Project Structure

agentic-rag  
│  
├── app  
│   ├── api        # FastAPI endpoints  
│   ├── agent      # agent workflow and nodes  
│   ├── rag        # retrieval pipeline  
│   ├── tools      # external tools  
│   └── core       # configuration  
│  
├── data           # documents for retrieval  
├── tests  
│  
├── Dockerfile  
├── docker-compose.yml  
└── requirements.txt  

The structure separates **API logic**, **agent workflow**, and **capability modules** to maintain a clean architecture.

---

## Development Approach

The development process follows several stages:

1. Establish containerized backend infrastructure
2. Implement graph-based agent workflow
3. Integrate retrieval-augmented generation
4. Add tool execution capability
5. Improve observability and evaluation

Each stage extends the reasoning capabilities of the agent while maintaining architectural clarity.

---

## Current Status

Early-stage development.

The current version includes:

- containerized development environment
- FastAPI backend service
- basic project structure
- vector database service

Agent workflow, retrieval logic, and tool execution will be implemented in later stages.

---

## License

MIT