import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- App Title ---
st.title("NSE Stock Price Predictor")

# --- Sidebar Inputs ---
stock = st.sidebar.text_input("Enter NSE Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
pred_days = st.sidebar.slider("Prediction Horizon (Days)", 1, 15, 5)
model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])

# --- Fetch Data ---
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

data = load_data(stock, start_date, end_date)

# --- Feature Engineering ---
def add_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26

# --- Prepare Data ---
data = add_indicators(data)
features = ['SMA_10', 'SMA_20', 'RSI', 'MACD']
X = data[features]
y = data['Close']

# --- Train-Test Split ---
train_size = int(len(data) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# --- Model Training ---
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

model.fit(X_train, y_train)

# --- Prediction ---
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# --- Forecasting Future Prices ---
def forecast_next_days(model, last_known_data, days):
    future_preds = []
    temp_data = last_known_data.copy()

    for _ in range(days):
        X_future = temp_data[features].iloc[-1:]
        next_pred = model.predict(X_future)[0]
        new_row = temp_data.iloc[-1].copy()
        new_row['Close'] = next_pred
        temp_data = pd.concat([temp_data, pd.DataFrame([new_row])], ignore_index=True)
        temp_data = add_indicators(temp_data)
        future_preds.append(next_pred)

    return future_preds

future_prices = forecast_next_days(model, data.copy(), pred_days)

# --- Plot Results ---
st.subheader("Model Backtesting")
fig, ax = plt.subplots()
ax.plot(y_test.index, y_test, label='Actual')
ax.plot(y_test.index, predictions, label='Predicted')
ax.set_title("Backtest: Actual vs Predicted")
ax.legend()
st.pyplot(fig)

st.write(f"Mean Squared Error: {mse:.2f}")

# --- Future Predictions ---
st.subheader(f"Forecast for Next {pred_days} Days")
future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=pred_days)
future_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_prices})
st.write(future_df.set_index("Date"))
