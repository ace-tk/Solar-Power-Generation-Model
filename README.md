# Solar Power Generation Forecasting

An AI/ML course project. Two parts:

- **Milestone 1** — short-term solar power forecasting using Linear Regression
  and Random Forest, with a Streamlit UI.
- **Milestone 2** — an agentic AI assistant on top of the forecaster that
  generates a structured grid optimization report. Built with LangGraph,
  FAISS retrieval over a curated corpus, and Google Gemini 2.5 Flash.
  PDF export and a chat interface for follow-up questions.

Authors: Ansh, Swarnim, Tanima, Tisha.

## Dataset

Anikannal Solar Power Generation Dataset, Plant 1
(https://www.kaggle.com/datasets/anikannal/solar-power-generation-data).

The generation file (`Plant_1_Generation_Data.csv`) is sampled at 15-minute
intervals; the weather sensor file (`Plant_1_Weather_Sensor_Data.csv`) is
hourly. We join the two with a time-aware merge and drop rows where AC
power is zero (night periods).

## Models and metrics

Both models use the same eight features: ambient temperature, module
temperature, irradiation, hour, month, day of week, lag-1 AC power, and a
3-period rolling mean of AC power. Train/test split is chronological 80/20.

| Model              | MAE (kW) | RMSE (kW) |
|--------------------|----------|-----------|
| Linear Regression  | 17.33    | 32.09     |
| Random Forest      | 17.26    | 32.45     |

Numbers are read from `models/metrics.json`, which the training script
writes.

## Project structure

```
.
├── app.py                   Streamlit app (both tabs)
├── requirements.txt
├── README.md
├── .env.example             template for GOOGLE_API_KEY
│
├── agent/                   Milestone 2 package
│   ├── schemas.py           Pydantic state and GridReport
│   ├── forecaster.py        24-hour iterative RF forecast
│   ├── rag.py               FAISS retrieval over knowledge/
│   ├── prompts.py           anti-hallucination prompts
│   ├── nodes.py             LangGraph node functions
│   ├── graph.py             StateGraph wiring
│   ├── chat.py              follow-up chat layer
│   └── pdf_export.py        reportlab PDF rendering
│
├── knowledge/               RAG corpus (Markdown, with [Source: ...] tags)
│   ├── grid_balancing.md
│   ├── storage_systems.md
│   ├── renewable_integration.md
│   └── demand_response.md
│
├── models/                  trained artifacts (.pkl, .json, .faiss)
├── scripts/train_model.py
└── docs/
    ├── report.tex           Milestone 1 report
    ├── report2.tex          Milestone 2 report
    └── references.bib
```

## Setup

```bash
git clone https://github.com/ace-tk/Solar-Power-Generation-Model.git
cd Solar-Power-Generation-Model
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey and put it
in a `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

For Streamlit Cloud, add the same key under App settings → Secrets.

The first tab (forecasting) does not need the key. Only the agent tab does.

## Running

```bash
streamlit run app.py
```

Open http://localhost:8501.

If you want to retrain the models from scratch (the trained `.pkl` files are
already committed):

```bash
python scripts/train_model.py
```

## Notes on the agent

The agent is a LangGraph state machine with five nodes:

1. `summarize_forecast` — natural-language summary of the day's curve.
2. `identify_variability` — flags low-output and high-ramp windows.
3. `retrieve_guidelines_node` — FAISS query over the knowledge corpus.
4. `draft_recommendations` — single Gemini 2.5 Flash call, schema-bound to
   the `GridReport` Pydantic model.
5. `format_structured_output` — passes the validated report through, or
   returns a guideline-only fallback if the LLM call fails.

The prompt forbids the model from inventing tariff values, vendor names, or
regulations not present in the inputs, and requires it to cite sources by
the names that appear in retrieved passages. Combined with the Pydantic
schema, this keeps the report's five sections grounded.

## Links

- Hosted app: https://solar-power-generation-model-4hhrmvg6yyguffdtcwghkz.streamlit.app
- Repository: https://github.com/ace-tk/Solar-Power-Generation-Model

## Deployment

The app is deployed on Streamlit Community Cloud. Push the repo to GitHub,
go to https://share.streamlit.io, point it at the repo, and add
`GOOGLE_API_KEY` under Secrets.
