"""Conversational layer over a generated GridReport.

The user can ask follow-up questions; we re-use the same RAG corpus and
Gemini model. Every response is grounded in (a) the forecast numbers,
(b) the structured report, and (c) freshly retrieved guidelines.
"""

from __future__ import annotations

from agent.nodes import _format_guidelines_block, _get_llm
from agent.rag import retrieve_guidelines
from agent.schemas import ForecastState, GridReport

CHAT_SYSTEM = """You are an analytical assistant answering follow-up questions
about a solar generation forecast and the grid optimization report already
produced for it.

Rules:
1. Ground every claim in the FORECAST, the REPORT, or the RETRIEVED_GUIDELINES.
   If the user asks something the inputs do not cover, say so plainly instead
   of inventing numbers, vendors, tariffs, or regulations.
2. Keep answers concise — typically 2-5 sentences. Use bullet lists only when
   the user asks for steps or a comparison.
3. Use kW for power and kWh for energy. Cite guideline sources by name when
   you draw on them.
"""


def _format_report(report: GridReport) -> str:
    parts = [
        f"FORECAST_SUMMARY: {report.forecast_summary}",
        f"VARIABILITY_AND_RISKS: {report.variability_and_risks}",
        "GRID_BALANCING_RECOMMENDATIONS:",
        *(f"  - {x}" for x in report.grid_balancing_recommendations),
        "STORAGE_RECOMMENDATIONS:",
        *(f"  - {x}" for x in report.storage_recommendations),
        "UTILIZATION_STRATEGIES:",
        *(f"  - {x}" for x in report.utilization_strategies),
        f"REFERENCES: {', '.join(report.references)}",
    ]
    return "\n".join(parts)


def _format_forecast(forecast: ForecastState) -> str:
    return (
        f"date={forecast.date}, peak={forecast.peak_kw:.1f} kW, "
        f"total={forecast.total_kwh:.1f} kWh, "
        f"low_power_hours={forecast.low_power_hours}, "
        f"high_variability_windows={forecast.high_variability_windows}"
    )


def chat_response(
    user_message: str,
    history: list[dict],
    forecast: ForecastState,
    report: GridReport,
) -> str:
    """Answer a follow-up question grounded in the forecast + report + RAG.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts
    representing the prior turns of this conversation.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    hits = retrieve_guidelines(user_message, k=4)
    context = (
        f"FORECAST: {_format_forecast(forecast)}\n\n"
        f"REPORT:\n{_format_report(report)}\n\n"
        f"RETRIEVED_GUIDELINES:\n{_format_guidelines_block(hits)}"
    )

    messages = [
        SystemMessage(content=CHAT_SYSTEM),
        SystemMessage(content=context),
    ]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=user_message))

    llm = _get_llm()
    try:
        result = llm.invoke(messages)
    except Exception as e:
        return f"(LLM call failed: {e}. Try again in a moment.)"
    return result.content if hasattr(result, "content") else str(result)
