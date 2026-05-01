"""LangChain Agentic RAG Pipeline using ReAct agent."""
from dataclasses import dataclass, field
from typing import Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.callbacks import LangChainTracer
from vectorstore.faiss_retriever import FAISSRetriever
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    answer: str
    sources: list[str]
    contexts: list[str]
    intermediate_steps: list = field(default_factory=list)


class AgenticRAGPipeline:
    """Production-grade agentic RAG pipeline with LangChain ReAct agent."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        index_path: str = "./data/faiss_index",
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.retriever = FAISSRetriever(index_path=index_path)
        self.agent_executor = self._build_agent()
        self._metrics = {"total_queries": 0, "avg_latency_ms": 0.0}

    def _build_agent(self) -> AgentExecutor:
        retrieval_tool = Tool(
            name="document_retriever",
            description="Search and retrieve relevant document chunks from the knowledge base. Input should be a search query.",
            func=self.retriever.retrieve_as_string,
        )

        prompt = PromptTemplate.from_template(
            """Answer the question using the document_retriever tool to find relevant context.

Question: {input}

{agent_scratchpad}

Thought: I should search the knowledge base for relevant information."""
        )

        agent = create_react_agent(self.llm, [retrieval_tool], prompt)
        return AgentExecutor(
            agent=agent,
            tools=[retrieval_tool],
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def run(self, query: str, top_k: int = 5) -> PipelineResult:
        """Execute the agentic RAG pipeline for a given query."""
        logger.info(f"Running pipeline for query: {query[:100]}")
        self.retriever.top_k = top_k

        result = self.agent_executor.invoke({"input": query})

        contexts = self.retriever.last_retrieved_chunks
        sources = list({chunk.metadata.get("source", "unknown") for chunk in self.retriever.last_retrieved_docs})

        self._metrics["total_queries"] += 1
        return PipelineResult(
            answer=result["output"],
            sources=sources,
            contexts=contexts,
            intermediate_steps=result.get("intermediate_steps", []),
        )

    def get_metrics(self) -> dict:
        return self._metrics
