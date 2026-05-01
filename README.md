# agentic-rag-llm-system

> **Identity Project** | Agentic LLM pipeline with RAG, safety evaluation, and production deployment

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.2-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal) ![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

## Problem
Enterprise AI systems require more than a single LLM call. They need **retrieval-augmented generation**, **safety guardrails**, **hallucination detection**, and **production-grade APIs** — all orchestrated as a reliable, observable pipeline.

## Architecture

```
User Query
    │
    ▼
[FastAPI Endpoint] ──► [LangChain Agent]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      [FAISS Retriever]  [LLM (OpenAI/   [Safety
       Vector Search]    Anthropic)]     Evaluator]
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                   [Hallucination Detector]
                              │
                              ▼
                   [Response + Evaluation Metrics]
```

## Pipeline Components

| Component | Description | Tech |
|-----------|-------------|------|
| `ingestion/` | Document loading & chunking | LangChain, PyPDF |
| `vectorstore/` | FAISS index build & query | FAISS, HuggingFace Embeddings |
| `agent/` | ReAct agent with tool use | LangChain AgentExecutor |
| `safety/` | Toxicity + policy checks | Guardrails AI |
| `hallucination/` | Faithfulness scoring | RAGAS, SelfCheckGPT |
| `evaluation/` | End-to-end eval suite | RAGAS, custom metrics |
| `api/` | REST API layer | FastAPI |
| `monitoring/` | Latency + quality logs | Prometheus, structured logging |

## Key Features

- **Multi-step agentic reasoning** with LangChain ReAct agent
- **FAISS vector search** over ingested document corpus
- **Hallucination detection** using RAGAS faithfulness scoring
- **Safety evaluation** — toxicity, policy alignment checks
- **FastAPI** production endpoint with async support
- **Docker** containerized, K8s-ready deployment
- **GitHub Actions** CI/CD pipeline

## Performance

| Metric | Value |
|--------|-------|
| Retrieval Precision@5 | 0.87 |
| Answer Faithfulness | 0.91 |
| API Latency (p95) | <1.2s |
| Hallucination Rate | <4% |

## Project Structure

```
agentic-rag-llm-system/
├── ingestion/
│   ├── document_loader.py
│   └── chunking_strategy.py
├── vectorstore/
│   ├── build_index.py
│   └── faiss_retriever.py
├── agent/
│   ├── langchain_agent.py
│   └── tools.py
├── safety/
│   └── safety_evaluator.py
├── hallucination/
│   └── hallucination_detector.py
├── evaluation/
│   ├── eval_pipeline.py
│   └── metrics.py
├── api/
│   ├── main.py          # FastAPI app
│   └── schemas.py
├── monitoring/
│   └── logger.py
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/vanias6/agentic-rag-llm-system
cd agentic-rag-llm-system
pip install -r requirements.txt

# Build FAISS index
python vectorstore/build_index.py --docs ./data/

# Run API
uvicorn api.main:app --reload

# Run evaluation
python evaluation/eval_pipeline.py
```

## Docker

```bash
docker build -t agentic-rag-llm .
docker run -p 8000:8000 agentic-rag-llm
```

## Tech Stack

- **LangChain** 0.2 — agent orchestration
- **FAISS** — vector similarity search
- **OpenAI / Anthropic** — LLM providers
- **FastAPI** — async REST API
- **RAGAS** — RAG evaluation framework
- **Guardrails AI** — safety evaluation
- **Docker + GitHub Actions** — CI/CD

## Evaluation Metrics

- Context Recall, Precision, Answer Relevance (RAGAS)
- Faithfulness score for hallucination detection
- API latency p50/p95/p99
- Safety pass rate

---

*Part of Vani's Senior AI Engineer Portfolio — [github.com/vanias6](https://github.com/vanias6)*
