import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------
# TECHNICAL INDICATORS FUNCTION
# --------------------------------------
def add_technical_indicators(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['UpperBB'], df['LowerBB'] = bollinger_bands(df['Close'])
    df['MACD'], df['MACD_signal'] = macd(df['Close'])
    df['OBV'] = obv(df['Close'], df['Volume'])
    df['ATR'] = atr(df['High'], df['Low'], df['Close'])
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['SuperTrend'] = df['Close']
    nine_high = df['High'].rolling(window=9).max()
    nine_low = df['Low'].rolling(window=9).min()
    df['Ichimoku_conversion'] = (nine_high + nine_low) / 2
    twenty_six_high = df['High'].rolling(window=26).max()
    twenty_six_low = df['Low'].rolling(window=26).min()
    df['Ichimoku_base'] = (twenty_six_high + twenty_six_low) / 2
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + 2*std, sma - 2*std

def macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal

def obv(close_series, volume_series):
    close = close_series.squeeze()
    volume = volume_series.squeeze()

    obv_values = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv_values.append(obv_values[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_values.append(obv_values[-1] - volume.iloc[i])
        else:
            obv_values.append(obv_values[-1])
    return pd.Series(obv_values, index=close.index)


def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# --------------------------------------
# APP START
# --------------------------------------
st.title("📈 Indian Stock Market Prediction & Analysis")

stocks = {
    'BAHART ELECTRONIC LTD': 'BEL.NS',
    'TATAMOTOR': 'TATAMOTORS.NS',
    'RAILTEL': 'RAILTEL.NS',
    'NATIONAL ALUMINIUM': 'NATIONALUM.NS',
    'RELIANCE': 'RELIANCE.NS',
    'TATA POWER': 'TATAPOWER.NS',
    'TCS': 'TCS.NS',
    'INFY': 'INFY.NS'
}
stock_name = st.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[stock_name]

period = st.slider("Select data period (days)", 30, 730, 180)
start_date = datetime.now() - timedelta(days=period)
end_date = datetime.now()

@st.cache_data

def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), interval='1d')
    if df.empty:
        st.error("No data found for the given ticker.")
        return df
    df = add_technical_indicators(df)
    return df

df = get_data(ticker, start_date, end_date)
if df.empty:
    st.stop()

st.subheader("Raw Data with Indicators")
st.dataframe(df.tail())

# Feature Engineering
feature_columns = [
    'MA20', 'MA50', 'RSI', 'UpperBB', 'LowerBB',
    'MACD', 'MACD_signal', 'OBV', 'ATR', 'VWAP',
    'Momentum', 'SuperTrend', 'Ichimoku_conversion', 'Ichimoku_base'
]
df.dropna(inplace=True)
features = df[feature_columns]
y = df['Close'].loc[features.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# BACKTESTING
model = XGBRegressor(n_estimators=100)
split = int(len(X_scaled) * 0.8)
model.fit(X_scaled[:split], y[:split])
y_pred = model.predict(X_scaled[split:])

# Ensure 1D arrays for metrics
actual = y[split:].values.flatten()
predicted = y_pred.flatten()

rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

# Backtesting Price Comparison Plot
st.write("### 📉 Actual vs Predicted Close Price")

plot_df = pd.DataFrame({
    'Actual': actual,
    'Predicted': predicted
}, index=y[split:].index)

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted Price'))
fig.update_layout(
    title="Actual vs Predicted Close Prices (Backtest)",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    legend=dict(x=0, y=1.0),
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig, use_container_width=True)
st.write("### 🔁 Backtesting Results")
st.write(f"**RMSE on Last {len(actual)} Days:** {rmse:.2f}")
st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:.2f}%")
st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# PREDICTION
X_last_raw = df[feature_columns].iloc[-1:]
X_last_scaled = scaler.transform(X_last_raw)
predicted_price = float(model.predict(X_last_scaled)[0])
latest_close = float(df['Close'].iloc[-1])

try:
    current_price = float(yf.Ticker(ticker).info['currentPrice'])
    is_live = True
except KeyError:
    current_price = latest_close
    is_live = False

st.markdown(f"""
### 📊 Current Price: ₹{current_price:.2f}
{"<span style='background-color:#00cc44;color:white;padding:4px 8px;border-radius:10px;font-size:0.8rem;'>LIVE</span>" if is_live else "<span style='background-color:#999;color:white;padding:4px 8px;border-radius:10px;font-size:0.8rem;'>CLOSE</span>"}
""", unsafe_allow_html=True)

st.markdown(f"""
<small>Yesterday's Close: ₹{latest_close:.2f}</small>
""", unsafe_allow_html=True)

recommendation = "BUY" if predicted_price > current_price else "SELL"
st.subheader(f"🔮 Predicted Next Close: ₹{predicted_price:.2f} → **{recommendation}**")

returns = df['Close'].pct_change().dropna()
volatility = float(returns.std() * np.sqrt(252))
sharpe_ratio = float(returns.mean() / returns.std())

st.write("### ⚠️ Risk Assessment")
st.write(f"**Annualized Volatility:** {volatility:.2%}")
st.write(f"**Sharpe Ratio (no risk-free adj.):** {sharpe_ratio:.2f}")
