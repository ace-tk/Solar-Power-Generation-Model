"""Prompt templates for the Grid Optimization agent.

All prompts include explicit anti-hallucination framing: the model is told
to stick to the provided forecast data and retrieved guidelines, and to
mark uncertainty rather than invent operational specifics.
"""

SYSTEM_PREAMBLE = """You are an analytical assistant for grid operators. You reason over
short-term solar generation forecasts and retrieved grid-management guidelines
to produce structured, actionable recommendations.

Rules you must follow without exception:
1. Ground every operational claim in either the forecast numbers provided or
   the retrieved guidelines. Do not invent tariff values, utility names,
   device specifications, or regulatory citations that are not in the inputs.
2. When forecast data is incomplete or uncertain, state the uncertainty
   explicitly rather than inventing a precise number.
3. Use the SI unit kW for power and kWh for energy throughout.
4. Cite guideline sources by name exactly as they appear in the retrieved
   passages.
"""


DRAFT_REPORT_PROMPT = """\
Using the forecast summary and retrieved guidelines below, produce a grid
optimization report with exactly these fields:

- forecast_summary: 2-3 sentences describing the day's expected generation
  shape, peak, and total energy. Reflect the numbers in FORECAST_STATS.
- variability_and_risks: 2-3 sentences identifying the hours or windows of
  greatest operational risk, drawing on FORECAST_STATS.low_power_hours and
  high_variability_windows.
- grid_balancing_recommendations: 3-5 bullet strings. Each should be concrete
  and reference a guideline or forecast number.
- storage_recommendations: 3-5 bullet strings on battery charge/discharge
  windows, SoC targets, and sizing relative to peak output.
- utilization_strategies: 3-5 bullet strings on demand-side actions, load
  shifting, and tariff or scheduling levers.
- references: list of source names actually used. Draw from the SOURCE field
  of each guideline you relied on. Do not invent sources.

FORECAST_STATS:
{forecast_stats}

VARIABILITY_NOTES:
{variability_notes}

RETRIEVED_GUIDELINES:
{guidelines}

Emit output that conforms exactly to the GridReport schema. If any field would
be speculative, write an explicit caveat ("confidence low") instead of making
up specifics.
"""


FORECAST_SUMMARY_PROMPT = """\
Summarize the following 24-hour solar generation forecast in 2-3 sentences.
Mention peak kW, total kWh, and the approximate production window. Use only
the numbers given; do not add context not supplied.

FORECAST_STATS:
{forecast_stats}
"""
