# ☀️ Solar Power Generation Forecasting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning-based solar power forecasting system that predicts AC power output using weather data and temporal patterns. Built with Streamlit for interactive predictions and model comparison.

## 🎯 Overview

This project implements time-series forecasting for solar power generation using the Anikannal Solar Power Plant Dataset. The system compares Linear Regression and Random Forest models to predict power output based on weather conditions and historical patterns.

### Key Features

- 🔮 **Real-time Predictions** - Interactive web interface for instant forecasting
- 📊 **Model Comparison** - Side-by-side evaluation of Linear Regression vs Random Forest
- 🌡️ **Weather Integration** - Incorporates temperature and irradiation data
- ⏰ **Temporal Analysis** - Captures daily and seasonal patterns
- 📈 **Visual Analytics** - Gauge charts and feature contribution plots

## 🚀 Live Demo

🌐 **[Try the App](https://your-app-url.streamlit.app)** *(Deploy and add your URL here)*

## 📊 Performance Metrics

| Model | MAE (kW) | RMSE (kW) | Error Rate |
|-------|----------|-----------|------------|
| Linear Regression | 17.33 | 32.09 | 1-5% |
| **Random Forest** | **17.26** | **32.45** | **1-5%** |

## 🏗️ Project Structure

```
solar-power-forecasting/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── models/                         # Trained ML models
│   ├── linear_model.pkl           # Linear Regression model
│   ├── random_forest_model.pkl    # Random Forest model (100 trees)
│   ├── scaler.pkl                 # StandardScaler for features
│   └── feature_list.pkl           # Feature names list
│
├── data/                           # Dataset files
│   ├── Plant_1_Generation_Data.csv    # 15-min generation data
│   └── Plant_1_Weather_Sensor_Data.csv # Hourly weather data
│
├── scripts/                        # Training scripts
│   └── train_model.py             # Model training pipeline
│
└── docs/                           # Documentation
    ├── DEPLOYMENT_PLAN.txt        # Deployment guide
    └── report.tex                 # LaTeX project report
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ace-tk/Solar-Power-Generation-Model.git
   cd Solar-Power-Generation-Model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models** (optional - pre-trained models included)
   ```bash
   python scripts/train_model.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`

## 📖 Usage

### Making Predictions

1. **Select Model**: Choose between Linear Regression or Random Forest
2. **Input Features**:
   - Weather: Ambient Temperature, Module Temperature, Irradiation
   - Time: Hour, Month, Day of Week
   - Historical: Previous AC Power, Rolling Mean
3. **Click Predict**: Get instant power output forecast
4. **View Results**: Gauge chart, model comparison, and feature analysis

### Sample Scenarios

**Peak Generation (Noon)**
```
Ambient Temp: 35°C
Module Temp: 45°C
Irradiation: 0.8 kW/m²
Hour: 12
Month: 5
```

**Morning Generation**
```
Ambient Temp: 25°C
Module Temp: 30°C
Irradiation: 0.4 kW/m²
Hour: 8
Month: 6
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Time-Aware Merge**: Used `pd.merge_asof()` to align 15-minute generation data with hourly weather data
- **Night Period Removal**: Filtered out records where AC_POWER = 0
- **Data Cleaning**: Removed duplicates and handled missing values

### 2. Feature Engineering

**Weather Features (3)**
- Ambient Temperature (°C)
- Module Temperature (°C)
- Irradiation (kW/m²)

**Temporal Features (3)**
- Hour (0-23) - Captures daily patterns
- Month (1-12) - Captures seasonal variations
- Day of Week (0-6) - Captures weekly trends

**Time-Series Features (2)**
- Lag-1: Previous AC power output
- Rolling Mean-3: 3-period moving average

### 3. Model Training

**Linear Regression**
- Baseline model with StandardScaler
- Fast training and inference
- Interpretable coefficients

**Random Forest**
- Ensemble of 100 decision trees
- Captures non-linear relationships
- Robust to outliers

### 4. Evaluation

- **Train-Test Split**: 80-20 chronological split (no shuffling)
- **Metrics**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
- **Cross-Validation**: Time-series aware validation

## 📊 Dataset

**Source**: Anikannal Solar Power Generation Dataset (Plant 1)

**Generation Data**
- Records: 34,000+
- Frequency: 15-minute intervals
- Features: DATE_TIME, AC_POWER, DC_POWER, DAILY_YIELD

**Weather Data**
- Records: 3,000+
- Frequency: Hourly measurements
- Features: AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

### Local Docker (Optional)

```bash
docker build -t solar-forecasting .
docker run -p 8501:8501 solar-forecasting
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Team Members**: Tanima, Tisha, Ansh, Swarnim
- **Course**: AI/ML Project
- **Institution**: [Your Institution Name]

## 🙏 Acknowledgments

- Anikannal Solar Power Plant for providing the dataset
- Streamlit for the amazing web framework
- scikit-learn for machine learning tools

## 📧 Contact

For questions or feedback, please reach out:
- GitHub: [@ace-tk](https://github.com/ace-tk)
- Email: your.email@example.com

## 🔗 Links

- [Live Demo](https://your-app-url.streamlit.app)
- [Documentation](docs/)
- [Project Report](docs/report.tex)
- [Dataset Source](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

---

⭐ **Star this repository if you found it helpful!**

Made with ❤️ for renewable energy forecasting
# Update
