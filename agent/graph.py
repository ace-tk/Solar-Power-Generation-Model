"""LangGraph state-graph wiring for the Grid Optimization agent."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    draft_recommendations,
    format_structured_output,
    identify_variability,
    retrieve_guidelines_node,
    summarize_forecast,
)
from agent.schemas import AgentState, ForecastState, GridReport


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("summarize", summarize_forecast)
    g.add_node("variability", identify_variability)
    g.add_node("retrieve", retrieve_guidelines_node)
    g.add_node("draft", draft_recommendations)
    g.add_node("format", format_structured_output)

    g.add_edge(START, "summarize")
    g.add_edge("summarize", "variability")
    g.add_edge("variability", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "format")
    g.add_edge("format", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()


def run_agent(forecast: ForecastState) -> GridReport:
    """Execute the full pipeline on a given forecast and return the report."""
    graph = get_compiled_graph()
    final_state = graph.invoke({"forecast": forecast})
    return final_state["report"]
