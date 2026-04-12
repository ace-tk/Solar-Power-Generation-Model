"""Render a GridReport (plus its ForecastState) to a styled PDF using reportlab."""

from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agent.schemas import ForecastState, GridReport


def _forecast_chart_png(forecast: ForecastState) -> BytesIO | None:
    """Return a PNG of the 24h forecast curve, or None if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    hours = [p.hour for p in forecast.points]
    values = [p.ac_power_kw for p in forecast.points]

    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=150)
    ax.plot(hours, values, color="#1f4e79", linewidth=2.0)
    ax.fill_between(hours, values, alpha=0.15, color="#1f4e79")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("24-hour Solar Generation Forecast")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def report_to_pdf(report: GridReport, forecast: ForecastState) -> bytes:
    """Render a GridReport + forecast chart to a PDF byte string."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Grid Optimization Report",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=colors.HexColor("#1f4e79"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#1f4e79"))
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Grid Optimization Report", h1))
    story.append(Paragraph(f"Forecast date: <b>{forecast.date}</b>", body))
    story.append(Spacer(1, 0.3 * cm))

    summary_table = Table(
        [
            ["Peak output", f"{forecast.peak_kw:.1f} kW"],
            ["Total energy", f"{forecast.total_kwh:.1f} kWh"],
            ["Low-output daytime hours", str(forecast.low_power_hours) if forecast.low_power_hours else "—"],
            ["High-variability windows", str(forecast.high_variability_windows) if forecast.high_variability_windows else "—"],
        ],
        colWidths=[5.5 * cm, 10 * cm],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c0d0e0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf1f8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.5 * cm))

    chart = _forecast_chart_png(forecast)
    if chart is not None:
        story.append(Image(chart, width=16 * cm, height=5.6 * cm))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Forecast summary", h2))
    story.append(Paragraph(report.forecast_summary, body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Variability and risk periods", h2))
    story.append(Paragraph(report.variability_and_risks, body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Grid balancing recommendations", h2))
    for item in report.grid_balancing_recommendations:
        story.append(Paragraph(f"• {item}", body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Storage recommendations", h2))
    for item in report.storage_recommendations:
        story.append(Paragraph(f"• {item}", body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Utilization strategies", h2))
    for item in report.utilization_strategies:
        story.append(Paragraph(f"• {item}", body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Supporting references", h2))
    for ref in report.references:
        story.append(Paragraph(f"• {ref}", body))

    doc.build(story)
    return buf.getvalue()
