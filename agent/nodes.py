"""LangGraph node functions for the Grid Optimization agent.

Each node takes the mutable AgentState, performs one step, and returns a
partial dict of new state. LangGraph merges the partials.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent.prompts import (
    DRAFT_REPORT_PROMPT,
    FORECAST_SUMMARY_PROMPT,
    SYSTEM_PREAMBLE,
)
from agent.rag import retrieve_guidelines
from agent.schemas import AgentState, GridReport


def _forecast_stats_for_prompt(forecast) -> str:
    """Compact JSON snapshot of the forecast, suitable for a prompt."""
    hourly = [
        {"hour": p.hour, "kw": round(p.ac_power_kw, 1), "irr": round(p.irradiation, 2)}
        for p in forecast.points
    ]
    return json.dumps(
        {
            "date": forecast.date,
            "peak_kw": round(forecast.peak_kw, 1),
            "total_kwh": round(forecast.total_kwh, 1),
            "low_power_hours": forecast.low_power_hours,
            "high_variability_windows": forecast.high_variability_windows,
            "hourly": hourly,
        },
        indent=2,
    )


def summarize_forecast(state: AgentState) -> dict[str, Any]:
    f = state["forecast"]
    daylight_hours = [p.hour for p in f.points if p.ac_power_kw > 0.05 * max(f.peak_kw, 1)]
    window = (
        f"{min(daylight_hours):02d}:00-{max(daylight_hours):02d}:00"
        if daylight_hours
        else "no significant daylight generation"
    )
    summary = (
        f"Forecast for {f.date}: peak {f.peak_kw:.0f} kW, total {f.total_kwh:.0f} kWh, "
        f"main generation window {window}."
    )
    return {"forecast_summary": summary}


def identify_variability(state: AgentState) -> dict[str, Any]:
    f = state["forecast"]
    notes: list[str] = []
    if f.low_power_hours:
        notes.append(
            f"Low-output daytime hours (below 20% of peak): {f.low_power_hours}."
        )
    if f.high_variability_windows:
        windows = ", ".join(f"{a:02d}:00-{b:02d}:00" for a, b in f.high_variability_windows)
        notes.append(f"High-variability windows (ramp > 30% of peak): {windows}.")
    if not notes:
        notes.append("No significant low-output or high-ramp windows detected.")
    return {"variability_summary": " ".join(notes)}


def retrieve_guidelines_node(state: AgentState) -> dict[str, Any]:
    f = state["forecast"]
    query = (
        f"Grid balancing and battery storage strategy for a solar plant with peak "
        f"{f.peak_kw:.0f} kW, total {f.total_kwh:.0f} kWh, with variability in hours "
        f"{f.low_power_hours or '—'} and ramp windows {f.high_variability_windows or '—'}."
    )
    hits = retrieve_guidelines(query, k=6)
    return {"guidelines": hits}


def _format_guidelines_block(hits) -> str:
    lines = []
    for h in hits:
        lines.append(f"- [{h.source}] {h.text}")
    return "\n".join(lines) if lines else "(no guidelines retrieved)"


def _get_llm():
    """Instantiate Gemini LLM. Reads key from env (populated at app entry)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Add it to .streamlit/secrets.toml locally "
            "or Streamlit Cloud Secrets in production."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key,
    )


def draft_recommendations(state: AgentState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = DRAFT_REPORT_PROMPT.format(
        forecast_stats=_forecast_stats_for_prompt(state["forecast"]),
        variability_notes=state.get("variability_summary", ""),
        guidelines=_format_guidelines_block(state.get("guidelines", [])),
    )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(GridReport)
    try:
        report: GridReport = structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PREAMBLE), HumanMessage(content=prompt)]
        )
    except Exception as e:
        return {
            "draft_report": _fallback_report(state),
            "error": f"LLM call failed: {e}. Falling back to guideline-only report.",
        }
    return {"draft_report": report}


def _fallback_report(state: AgentState) -> GridReport:
    """Guideline-only report, used when the LLM call fails."""
    f = state["forecast"]
    hits = state.get("guidelines", [])
    refs = sorted({h.source for h in hits}) or ["IEA Renewables Integration 2022"]
    return GridReport(
        forecast_summary=state.get(
            "forecast_summary",
            f"Forecast for {f.date}: peak {f.peak_kw:.0f} kW, total {f.total_kwh:.0f} kWh.",
        ),
        variability_and_risks=state.get("variability_summary", "No variability notes."),
        grid_balancing_recommendations=[
            "Hold spinning reserves of 3-7% of forecast solar during daylight hours "
            "(confidence low: LLM unavailable, rule-of-thumb from guidelines)."
        ],
        storage_recommendations=[
            "Target charge window 10:00-14:00, discharge 17:00-21:00; cap SoC at 80% "
            "to preserve cycle life (confidence low: LLM unavailable)."
        ],
        utilization_strategies=[
            "Signal demand-response aggregators day-ahead so flexible load can shift "
            "into midday surplus hours (confidence low: LLM unavailable)."
        ],
        references=refs,
    )


def format_structured_output(state: AgentState) -> dict[str, Any]:
    draft = state.get("draft_report")
    if draft is None:
        return {"report": _fallback_report(state)}
    # Already a validated GridReport from with_structured_output; pass through.
    return {"report": draft}
