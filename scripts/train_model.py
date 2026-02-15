import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Load datasets
try:
    gen_df = pd.read_csv('data/Plant_1_Generation_Data.csv')
    weather_df = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')
except FileNotFoundError as e:
    print(f"Error: Required CSV file not found - {e}")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Convert to datetime
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])

# Sort by time
gen_df = gen_df.sort_values('DATE_TIME')
weather_df = weather_df.sort_values('DATE_TIME')

# Merge using merge_asof
merged_df = pd.merge_asof(gen_df, weather_df, on='DATE_TIME', direction='nearest')

# Remove zero AC power (night periods)
merged_df = merged_df[merged_df['AC_POWER'] > 0]

# Remove duplicates
merged_df = merged_df.drop_duplicates()

# Feature engineering
merged_df['HOUR'] = merged_df['DATE_TIME'].dt.hour
merged_df['MONTH'] = merged_df['DATE_TIME'].dt.month
merged_df['DAY_OF_WEEK'] = merged_df['DATE_TIME'].dt.dayofweek

# Temporal features
merged_df['lag_1'] = merged_df['AC_POWER'].shift(1)
merged_df['rolling_mean_3'] = merged_df['AC_POWER'].rolling(window=3).mean()

# Drop NaN
merged_df = merged_df.dropna()

# Select features
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
            'HOUR', 'MONTH', 'DAY_OF_WEEK', 'lag_1', 'rolling_mean_3']
X = merged_df[features]
y = merged_df['AC_POWER']

# Time-series split (no shuffle)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression (with scaling)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
print(f"Linear Regression - MAE: {mean_absolute_error(y_test, lr_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"Random Forest - MAE: {mean_absolute_error(y_test, rf_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")

# Save models
try:
    with open('models/linear_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/feature_list.pkl', 'wb') as f:
        pickle.dump(features, f)
    print("\nModels saved successfully!")
except Exception as e:
    print(f"Error saving models: {e}")
    exit(1)
