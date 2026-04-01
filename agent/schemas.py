"""Typed state and output contracts for the Grid Optimization agent."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    hour: int = Field(ge=0, le=23)
    ac_power_kw: float = Field(ge=0.0)
    irradiation: float
    module_temp: float


class ForecastState(BaseModel):
    date: str
    points: list[ForecastPoint]
    peak_kw: float
    total_kwh: float
    low_power_hours: list[int] = Field(default_factory=list)
    high_variability_windows: list[tuple[int, int]] = Field(default_factory=list)


class GridReport(BaseModel):
    forecast_summary: str
    variability_and_risks: str
    grid_balancing_recommendations: list[str]
    storage_recommendations: list[str]
    utilization_strategies: list[str]
    references: list[str]


class RetrievedGuideline(BaseModel):
    text: str
    source: str
    score: float


class AgentState(TypedDict, total=False):
    """LangGraph mutable state. Fields are filled in progressively by nodes."""

    forecast: ForecastState
    forecast_summary: str
    variability_summary: str
    guidelines: list[RetrievedGuideline]
    draft_report: GridReport
    report: GridReport
    error: str
