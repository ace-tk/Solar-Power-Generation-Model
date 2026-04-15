import json
import os
import pickle
from datetime import date as _date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(page_title="Solar Power Forecasting", page_icon="🔆", layout="wide")


# Gemini API key: Streamlit Cloud secrets first, then local env / .env
def _configure_api_key() -> bool:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
            return True
    except Exception:
        pass  # No secrets file locally — fall through to .env
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return bool(os.environ.get("GOOGLE_API_KEY"))


_GEMINI_CONFIGURED = _configure_api_key()


# Load models
@st.cache_resource
def load_models_v2():
    try:
        with open('models/linear_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/feature_list.pkl', 'rb') as f:
            features = pickle.load(f)
        try:
            with open('models/metrics.json', 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = None
        return lr_model, scaler, rf_model, features, metrics
    except FileNotFoundError as e:
        st.error(f" Models not found: {e}. Please run 'python scripts/train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f" Error loading models: {e}")
        st.stop()

lr_model, scaler, rf_model, features, metrics = load_models_v2()

# Title
st.title("Solar Power Generation Forecasting")
st.markdown("Predict solar power output and generate grid-optimization reports")

# Model Performance Metrics
with st.expander("Model Performance Comparison"):
    if metrics is not None:
        lr_mae = metrics["linear_regression"]["mae"]
        lr_rmse = metrics["linear_regression"]["rmse"]
        rf_mae = metrics["random_forest"]["mae"]
        rf_rmse = metrics["random_forest"]["rmse"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Linear Regression MAE", f"{lr_mae:.2f} kW")
            st.metric("Linear Regression RMSE", f"{lr_rmse:.2f} kW")
        with col2:
            st.metric("Random Forest MAE", f"{rf_mae:.2f} kW",
                      delta=f"{rf_mae - lr_mae:+.2f}", delta_color="inverse")
            st.metric("Random Forest RMSE", f"{rf_rmse:.2f} kW",
                      delta=f"{rf_rmse - lr_rmse:+.2f}", delta_color="inverse")
    else:
        st.info("Run `python scripts/train_model.py` to generate `models/metrics.json`.")

# Sidebar - Model Selection
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Linear Regression"])

# Sidebar - About
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app predicts solar power generation using:\n\n"
    "- Weather data (temp, irradiation)\n"
    "- Temporal features (hour, month)\n"
    "- Historical patterns (lag, rolling mean)\n\n"
    "**Dataset:** Anikannal Solar Plant 1"
)

# Sidebar - Quick Tips
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Tips")
st.sidebar.markdown(
    "**High irradiation** (>0.5 kW/m²) = Higher power\n\n"
    "**Peak hours** (10-14) = Maximum generation\n\n"
    "**Module temp** affects efficiency"
)

# Sidebar - Agent status (Milestone 2)
st.sidebar.markdown("---")
st.sidebar.subheader("Grid Optimization Agent")
if _GEMINI_CONFIGURED:
    st.sidebar.success("Gemini API key detected")
else:
    st.sidebar.warning(
        "Gemini API key not set. Add `GOOGLE_API_KEY` to Streamlit secrets "
        "(or a local `.env`) to enable the agent."
    )


tab1, tab2 = st.tabs(["Forecasting", "Grid Optimization Agent"])

with tab1:
    # Input Section
    st.header("Input Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        module_temp = st.number_input("Module Temperature (°C)", min_value=0.0, max_value=70.0, value=35.0, step=0.1)
        irradiation = st.number_input("Irradiation (kW/m²)", min_value=0.0, max_value=1.5, value=0.5, step=0.01)

    with col2:
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12)
        month = st.slider("Month", min_value=1, max_value=12, value=6)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=3)

    with col3:
        lag_1 = st.number_input("Previous AC Power (kW)", min_value=0.0, max_value=1500.0, value=300.0, step=10.0)
        rolling_mean_3 = st.number_input("Rolling Mean (3 periods)", min_value=0.0, max_value=1500.0, value=300.0, step=10.0)

    # Predict Button
    if st.button("Predict Power Output", type="primary"):
        input_data = np.array([[ambient_temp, module_temp, irradiation, hour, month, day_of_week, lag_1, rolling_mean_3]])

        input_scaled = scaler.transform(input_data)
        lr_pred = lr_model.predict(input_scaled)[0]
        rf_pred = rf_model.predict(input_data)[0]

        prediction = lr_pred if model_choice == "Linear Regression" else rf_pred

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            title={'text': "Predicted AC Power (kW)"},
            delta={'reference': lag_1},
            gauge={'axis': {'range': [0, 1500]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 500], 'color': "lightgray"},
                       {'range': [500, 1000], 'color': "gray"},
                       {'range': [1000, 1500], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1200}}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Predictions")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Linear Regression", f"{lr_pred:.2f} kW")
        with col_b:
            st.metric("Random Forest", f"{rf_pred:.2f} kW")
        with col_c:
            diff = abs(lr_pred - rf_pred)
            st.metric("Difference", f"{diff:.2f} kW")

        st.subheader("Input Feature Values")
        feature_df = pd.DataFrame({
            'Feature': features,
            'Value': [ambient_temp, module_temp, irradiation, hour, month, day_of_week, lag_1, rolling_mean_3]
        })
        fig2 = px.bar(feature_df, x='Feature', y='Value', color='Value',
                      color_continuous_scale='Blues', title="Feature Values")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Sample Scenarios
    with st.expander("Sample Scenarios - Try These!"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Peak Generation**")
            st.code("""
Ambient Temp: 35°C
Module Temp: 45°C
Irradiation: 0.8 kW/m²
Hour: 12
Month: 5
Lag: 800 kW
Rolling Mean: 750 kW
""")

        with col2:
            st.markdown("**Morning**")
            st.code("""
Ambient Temp: 25°C
Module Temp: 30°C
Irradiation: 0.4 kW/m²
Hour: 8
Month: 6
Lag: 400 kW
Rolling Mean: 350 kW
""")

        with col3:
            st.markdown("**Evening**")
            st.code("""
Ambient Temp: 28°C
Module Temp: 35°C
Irradiation: 0.3 kW/m²
Hour: 17
Month: 7
Lag: 200 kW
Rolling Mean: 250 kW
""")


with tab2:
    st.header("Grid Optimization Agent")
    st.markdown(
        "Generates a 24-hour solar forecast, retrieves grid-management guidelines, "
        "and drafts a structured optimization report using a LangGraph + Gemini agent."
    )

    if not _GEMINI_CONFIGURED:
        st.error(
            "Gemini API key not configured. Set `GOOGLE_API_KEY` in "
            "`.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets (deploy), "
            "then reload."
        )

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        forecast_date = st.date_input("Forecast date", value=_date.today())
    with ac2:
        weather_pattern = st.selectbox(
            "Expected weather",
            options=["sunny", "partly_cloudy", "overcast"],
            format_func=lambda s: s.replace("_", " ").title(),
        )
    with ac3:
        ambient_c = st.slider("Ambient temperature (°C)", 10.0, 45.0, 28.0, 0.5)

    seed_lag = st.slider(
        "Seed lag (prior hour AC power, kW)",
        0.0, 1500.0, 0.0, 10.0,
        help="Used to bootstrap the iterative forecast. 0 is fine for an early-morning start.",
    )

    if "m2_report" not in st.session_state:
        st.session_state.m2_report = None
        st.session_state.m2_forecast = None
    if "m2_chat" not in st.session_state:
        st.session_state.m2_chat = []

    run_btn = st.button(
        "Generate Grid Optimization Report",
        type="primary",
        disabled=not _GEMINI_CONFIGURED,
    )

    if run_btn:
        from agent.forecaster import generate_24h_forecast
        from agent.graph import run_agent

        with st.status("Running agent pipeline...", expanded=True) as status:
            status.write("Generating 24-hour forecast curve...")
            forecast = generate_24h_forecast(
                date=forecast_date.isoformat(),
                pattern=weather_pattern,
                ambient_temp=float(ambient_c),
                seed_lag=float(seed_lag),
            )
            status.write(
                f"Peak: {forecast.peak_kw:.1f} kW · Total: {forecast.total_kwh:.1f} kWh"
            )

            status.write("Retrieving grid-management guidelines and calling Gemini...")
            report = run_agent(forecast)

            status.update(label="Report ready", state="complete", expanded=False)

        st.session_state.m2_report = report
        st.session_state.m2_forecast = forecast
        st.session_state.m2_chat = []

    report = st.session_state.m2_report
    forecast = st.session_state.m2_forecast

    if report is not None and forecast is not None:
        st.subheader("24-hour forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[p.hour for p in forecast.points],
            y=[p.ac_power_kw for p in forecast.points],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#1f4e79", width=2),
            name="AC Power (kW)",
        ))
        fig.update_layout(
            xaxis_title="Hour of day",
            yaxis_title="AC Power (kW)",
            height=300,
            margin=dict(l=20, r=20, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        cA, cB, cC = st.columns(3)
        cA.metric("Peak", f"{forecast.peak_kw:.0f} kW")
        cB.metric("Total", f"{forecast.total_kwh:.0f} kWh")
        cC.metric("Low-output hours", str(forecast.low_power_hours) if forecast.low_power_hours else "—")

        st.subheader("Forecast summary")
        st.write(report.forecast_summary)

        st.subheader("Variability and risk periods")
        st.write(report.variability_and_risks)

        st.subheader("Grid balancing recommendations")
        for item in report.grid_balancing_recommendations:
            st.markdown(f"- {item}")

        st.subheader("Storage recommendations")
        for item in report.storage_recommendations:
            st.markdown(f"- {item}")

        st.subheader("Utilization strategies")
        for item in report.utilization_strategies:
            st.markdown(f"- {item}")

        st.subheader("Supporting references")
        for ref in report.references:
            st.markdown(f"- {ref}")

        from agent.pdf_export import report_to_pdf

        pdf_bytes = report_to_pdf(report, forecast)
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"grid_optimization_{forecast.date}.pdf",
            mime="application/pdf",
            type="primary",
        )

        st.markdown("---")
        st.subheader("Ask the agent")
        st.caption(
            "Follow-up questions are answered using the forecast, the report above, "
            "and the same RAG corpus. The agent will say so if a question is outside scope."
        )

        chat_container = st.container()
        with chat_container:
            for turn in st.session_state.m2_chat:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

        if st.session_state.m2_chat and st.button("Clear chat"):
            st.session_state.m2_chat = []
            st.rerun()

        user_msg = st.chat_input("e.g. Why charge between 10:00 and 14:00?")
        if user_msg:
            from agent.chat import chat_response

            with st.spinner("Thinking..."):
                reply = chat_response(
                    user_message=user_msg,
                    history=st.session_state.m2_chat,
                    forecast=forecast,
                    report=report,
                )
            st.session_state.m2_chat.append({"role": "user", "content": user_msg})
            st.session_state.m2_chat.append({"role": "assistant", "content": reply})
            st.rerun()


st.markdown("---")
st.caption("Solar Power Forecasting System | Built with Streamlit")
