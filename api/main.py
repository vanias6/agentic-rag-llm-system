"""FastAPI entry point for the Agentic RAG LLM System."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import logging

from agent.langchain_agent import AgenticRAGPipeline
from hallucination.hallucination_detector import HallucinationDetector
from safety.safety_evaluator import SafetyEvaluator
from monitoring.logger import get_logger

app = FastAPI(
    title="Agentic RAG LLM System",
    description="Production-grade agentic pipeline with RAG, safety evaluation, and hallucination detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)
pipeline = AgenticRAGPipeline()
hallucination_detector = HallucinationDetector()
safety_evaluator = SafetyEvaluator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_safety_check: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    hallucination_score: float
    safety_passed: bool
    latency_ms: float
    faithfulness: float


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main RAG query endpoint with safety and hallucination evaluation."""
    start = time.time()
    logger.info(f"Received query: {request.query[:100]}")

    if request.enable_safety_check:
        safety_result = safety_evaluator.evaluate(request.query)
        if not safety_result.passed:
            raise HTTPException(status_code=400, detail=f"Safety check failed: {safety_result.reason}")

    result = pipeline.run(query=request.query, top_k=request.top_k)

    hallucination_result = hallucination_detector.score(
        question=request.query,
        answer=result.answer,
        contexts=result.contexts,
    )

    latency_ms = (time.time() - start) * 1000
    logger.info(f"Query completed in {latency_ms:.1f}ms | faithfulness={hallucination_result.faithfulness:.3f}")

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        hallucination_score=hallucination_result.score,
        safety_passed=True,
        latency_ms=latency_ms,
        faithfulness=hallucination_result.faithfulness,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metrics")
async def metrics():
    """Expose evaluation metrics endpoint."""
    return pipeline.get_metrics()
