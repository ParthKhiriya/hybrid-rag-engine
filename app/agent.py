import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.llm import LLMProvider
from app.models import QueryResponse, SourceResult
from app.observability import END_TO_END_LATENCY, GENERATION_LATENCY, RETRIEVAL_LATENCY
from app.retriever import HybridRetriever
from app.security import validate_output


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    sources: list[SourceResult]
    entities: list[str]
    answer: str
    output_flagged: bool


class RAGAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider) -> None:
        self.retriever = retriever
        self.llm = llm
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        self.graph = graph.compile()

    def _retrieve(self, state: AgentState) -> AgentState:
        with RETRIEVAL_LATENCY.time():
            result = self.retriever.retrieve(state["question"], state.get("top_k", 5))
        return {"sources": result.sources, "entities": result.entities}

    def _generate(self, state: AgentState) -> AgentState:
        with GENERATION_LATENCY.labels(provider=self.llm.name).time():
            answer = self.llm.answer(state["question"], state.get("sources", []))
        scan = validate_output(answer)
        return {"answer": answer, "output_flagged": scan.flagged}

    def invoke(self, question: str, top_k: int = 5) -> QueryResponse:
        started = time.perf_counter()
        with END_TO_END_LATENCY.time():
            result = self.graph.invoke({"question": question, "top_k": top_k})
        latency_ms = (time.perf_counter() - started) * 1000
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            entities=result.get("entities", []),
            latency_ms=round(latency_ms, 2),
            provider=self.llm.name,
            output_flagged=result.get("output_flagged", False),
        )
