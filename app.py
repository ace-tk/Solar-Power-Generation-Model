import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="Solar Power Forecasting", page_icon="🔆", layout="wide")

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
        return lr_model, scaler, rf_model, features
    except FileNotFoundError as e:
        st.error(f" Models not found: {e}. Please run 'python train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f" Error loading models: {e}")
        st.stop()

lr_model, scaler, rf_model, features = load_models_v2()

# Title
st.title("Solar Power Generation Forecasting")
st.markdown("Predict solar power output using weather and temporal features")

# Model Performance Metrics
with st.expander("Model Performance Comparison"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Linear Regression MAE", "16.80 kW")
        st.metric("Linear Regression RMSE", "32.10 kW")
    with col2:
        st.metric("Random Forest MAE", "16.75 kW", delta="-0.05", delta_color="inverse")
        st.metric("Random Forest RMSE", "32.04 kW", delta="-0.06", delta_color="inverse")

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
    # Prepare input
    input_data = np.array([[ambient_temp, module_temp, irradiation, hour, month, day_of_week, lag_1, rolling_mean_3]])
    
    # Predict with both models
    input_scaled = scaler.transform(input_data)
    lr_pred = lr_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_data)[0]
    
    # Select model
    prediction = lr_pred if model_choice == "Linear Regression" else rf_pred
    
    # Gauge Chart
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
    
    # Model Comparison
    st.subheader("Model Predictions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Linear Regression", f"{lr_pred:.2f} kW")
    with col_b:
        st.metric("Random Forest", f"{rf_pred:.2f} kW")
    with col_c:
        diff = abs(lr_pred - rf_pred)
        st.metric("Difference", f"{diff:.2f} kW")
    
    # Feature Contribution
    st.subheader("Input Feature Values")
    feature_df = pd.DataFrame({
        'Feature': features,
        'Value': [ambient_temp, module_temp, irradiation, hour, month, day_of_week, lag_1, rolling_mean_3]
    })
    fig2 = px.bar(feature_df, x='Feature', y='Value', color='Value', 
                  color_continuous_scale='Blues', title="Feature Values")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

# Footer
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

st.caption("Solar Power Forecasting System | Built with Streamlit")
