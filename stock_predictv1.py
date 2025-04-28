import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import plotly.graph_objects as go
import plotly.express as px
import requests
import urllib.parse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet
from transformers import pipeline
import datetime
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pytz


warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(layout="wide", page_title="Advanced Stock OHLC Prediction")

# Streamlit App Title with enhanced styling
st.markdown(
    f'<h1 style="font-size: 36px; color: #1C39BB; text-align: center;">📈 Advanced Stock OHLC Prediction App by Fayyaz</h1>',
    unsafe_allow_html=True)
st.markdown(
    f'<p style="font-size: 16px; color: #666; text-align: center;">Multi-model ensemble forecasting with advanced technical indicators and sentiment analysis.</p>',
    unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["Prediction & Analysis", "Market Sentiment", "Advanced Indicators", "Risk Assessment"])

with tab1:
    # Select stock and date range
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        stock_symbol = st.text_input("Enter NSE stock symbol (e.g. BEL.NS):", "BEL.NS")
    with col2:
        start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    with col3:
        end_date = st.date_input("End Date", pd.to_datetime("today"))

    if start_date >= end_date:
        st.warning("⚠️ End date must be after start date.")
        st.stop()


    # Download historical data
    @st.cache_data(ttl=3600)
    def get_stock_data(symbol, start, end):
        try:
            data = yf.download(symbol, start=start, end=end + pd.Timedelta(days=1))
            if data.empty:
                st.error(f"No data found for {symbol}. Please check the stock symbol.")
                st.stop()
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()


    raw_df = get_stock_data(stock_symbol, start_date, end_date)

    # Show basic stock info
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">Stock Data for {stock_symbol}</h2>',
                unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = float(raw_df['Close'].iloc[-1])  # Explicit conversion to float
        prev_price = float(raw_df['Close'].iloc[-2])  # Explicit conversion to float
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        st.metric("Current Price", f"₹{current_price:.2f}", f"{change_pct:.2f}%")

    with col2:
        high_52w = raw_df['High'].iloc[-252:].max() if len(raw_df) > 252 else raw_df['High'].max()
        high_52w = float(high_52w)  # Convert to float
        pct_from_high = ((current_price / high_52w) - 1) * 100
        st.metric("52-Week High", f"₹{high_52w:.2f}", f"{pct_from_high:.2f}%")
    with col3:
        low_52w = raw_df['Low'].iloc[-252:].min() if len(raw_df) > 252 else raw_df['Low'].min()
        low_52w = float(low_52w)  # Convert to float
        pct_from_low = ((current_price / low_52w) - 1) * 100
        st.metric("52-Week Low", f"₹{low_52w:.2f}", f"{pct_from_low:.2f}%")
    with col4:
        avg_vol = float(raw_df['Volume'].mean())
        recent_vol = float(raw_df['Volume'].iloc[-5:].mean())
        vol_change = ((recent_vol / avg_vol) - 1) * 100
        st.metric("Avg Volume", f"{int(avg_vol):,}", f"{vol_change:.2f}%")
    with st.expander("View Raw Data"):
        st.dataframe(raw_df.tail())


    # ----------------------------- Feature Engineering -----------------------------
    @st.cache_data(ttl=3600)
    def create_features(df):
        # For time series data where index is dates
        data = df.copy()
        data = data.loc[~data.index.duplicated(keep='first')]

        # Manually calculate MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']

        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low14 = data['Low'].rolling(window=14).min()
        high14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = 100 * (data['Close'] - low14) / (high14 - low14)
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

        # Bollinger Bands manual calculation
        sma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        data['Bollinger_Upper'] = sma20 + (std20 * 2)
        data['Bollinger_Lower'] = sma20 - (std20 * 2)

        # Trend calculation - FIXED VERSION
        close_values = data['Close'].to_numpy().flatten()
        upper_values = data['Bollinger_Upper'].to_numpy().flatten()
        lower_values = data['Bollinger_Lower'].to_numpy().flatten()

        data['Trend_Bollinger'] = np.select(
            [
                close_values > upper_values,
                close_values < lower_values
            ],
            [1, 0],  # 1 = Uptrend, 0 = Downtrend
            default=1
        )

        # Moving Averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        # Returns and Volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=5).std()

        data.dropna(inplace=True)
        return data


    # Process features
    with st.spinner("Processing technical indicators..."):
        df = create_features(raw_df)

    # ----------------------------- Machine Learning Models -----------------------------
    # Define features for ML
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 'Returns',
                       'Volatility', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Stoch_K',
                       'Stoch_D', 'ATR', 'OBV', 'MFI', 'Gap_Percentage', 'Daily_Range',
                       'Range_Percentage', 'Price_Momentum', 'Relative_Volume', 'Day_of_Week', 'Month']
    # Remove any features with NaN values using safe conversion
    valid_features = [
        col for col in feature_columns
        if col in df.columns
           and not df[col].isna().any().item()  # Convert to native bool
    ]

    X = df[valid_features]

    # Targets: Open, High, Low, Close
    ohlc_targets = ['Open', 'High', 'Low', 'Close']
    y_ohlc = df[ohlc_targets].loc[X.index]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series split
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_ohlc.iloc[:split], y_ohlc.iloc[split:]


    # ----- XGBoost Ensemble Models -----
    @st.cache_resource
    def train_stacking_model(X_train, y_train):
        try:
            # Define base models
            base_models = [
                ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
                ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
            ]

            # Meta learner
            meta_model = Ridge()

            # Create stacking ensemble
            stacking_model = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5
            )

            # Create multi-output version
            multi_model = MultiOutputRegressor(stacking_model)

            # Fit model
            multi_model.fit(X_train, y_train)

            return multi_model
        except Exception as e:
            st.error(f"Error training stacking model: {e}")
            # Fallback to XGBoost
            model = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42))
            model.fit(X_train, y_train)
            return model


    # ----- LSTM Model -----
    @st.cache_resource
    def train_lstm_model(df):
        try:
            # Prepare data for LSTM
            close_scaler = MinMaxScaler()
            scaled_close = close_scaler.fit_transform(df[['Close']])

            # Create sequences
            def create_sequences(data, seq_length):
                xs, ys = [], []
                for i in range(len(data) - seq_length):
                    x = data[i:i + seq_length]
                    y = data[i + seq_length]
                    xs.append(x)
                    ys.append(y)
                return np.array(xs), np.array(ys)

            seq_length = 20  # Look back 20 days
            X_lstm, y_lstm = create_sequences(scaled_close, seq_length)

            # Split data
            split_idx = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train with early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            model.fit(
                X_train_lstm, y_train_lstm,
                epochs=30,
                batch_size=32,
                validation_data=(X_test_lstm, y_test_lstm),
                callbacks=[early_stop],
                verbose=0
            )

            return model, close_scaler, seq_length
        except Exception as e:
            st.error(f"Error training LSTM model: {e}")
            return None, None, None


    # ----- Prophet Model -----
    @st.cache_resource
    def train_prophet_model(df):
        try:
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close'].values.ravel()  # Convert to 1D array
            })
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.add_country_holidays(country_name='IN')
            m.fit(prophet_df)
            return m
        except Exception as e:
            st.error(f"Error training Prophet model: {e}")
            return None


    # Train all models
    with st.spinner("Training machine learning models..."):
        # Train stacking model
        stacking_model = train_stacking_model(X_train, y_train)

        # Train LSTM model
        lstm_model, close_scaler, seq_length = train_lstm_model(df)

        # Train Prophet model
        prophet_model = train_prophet_model(df)

    # Make predictions with each model
    # Stacking model prediction
    y_ohlc_pred = stacking_model.predict(X_test)

    # LSTM prediction
    if lstm_model is not None and close_scaler is not None:
        last_sequence = df['Close'].iloc[-seq_length:].values.reshape(-1, 1)
        last_sequence_scaled = close_scaler.transform(last_sequence)
        last_sequence_reshaped = last_sequence_scaled.reshape(1, seq_length, 1)
        next_day_scaled = lstm_model.predict(last_sequence_reshaped)
        lstm_next_day_price = close_scaler.inverse_transform(next_day_scaled)[0][0]
    else:
        lstm_next_day_price = None

    # Prophet prediction
    if prophet_model is not None:
        future = prophet_model.make_future_dataframe(periods=7)  # 7-day forecast
        forecast = prophet_model.predict(future)
        prophet_next_day_price = forecast['yhat'].iloc[-1]
    else:
        prophet_next_day_price = None

    # Predict next day's OHLC using stacking model
    X_last_scaled = scaler.transform(df[valid_features].iloc[[-1]])
    predicted_ohlc = [float(x) for x in stacking_model.predict(X_last_scaled)[0]]


    # Ensemble the predictions
    def ensemble_predictions(xgb_pred, lstm_pred, prophet_pred):
        """Combine predictions from multiple models with dynamic weighting"""

        # Set weights
        models_available = sum([pred is not None for pred in [xgb_pred, lstm_pred, prophet_pred]])

        if models_available == 1:
            # Only one model available
            if xgb_pred is not None:
                return xgb_pred, {"XGBoost": 1.0}
            elif lstm_pred is not None:
                return lstm_pred, {"LSTM": 1.0}
            else:
                return prophet_pred, {"Prophet": 1.0}

        # Calculate weights based on availability
        weights = {}
        if xgb_pred is not None:
            weights["XGBoost"] = 0.5
        if lstm_pred is not None:
            weights["LSTM"] = 0.3
        if prophet_pred is not None:
            weights["Prophet"] = 0.2

        # Normalize weights
        total_weight = sum(weights.values())
        for k in weights:
            weights[k] /= total_weight

        # Combine predictions
        ensemble_pred = 0
        if xgb_pred is not None:
            ensemble_pred += xgb_pred * weights["XGBoost"]
        if lstm_pred is not None:
            ensemble_pred += lstm_pred * weights["LSTM"]
        if prophet_pred is not None:
            ensemble_pred += prophet_pred * weights["Prophet"]

        return ensemble_pred, weights


    # Create ensemble prediction for close price
    ensemble_close, weights = ensemble_predictions(
        xgb_pred=float(predicted_ohlc[3]),  # Close price from XGBoost
        lstm_pred=float(lstm_next_day_price) if lstm_next_day_price is not None else None,
        prophet_pred=float(prophet_next_day_price) if prophet_next_day_price is not None else None
    )

    # Yesterday's OHLC
    yesterday_ohlc = [float(x) for x in raw_df[ohlc_targets].iloc[-2].values] if len(raw_df) > 2 else None

    # Latest OHLC (today's actual)
    today_ohlc = [float(x) for x in raw_df[ohlc_targets].iloc[-1].values]

    # Display comparison table
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🔍 OHLC Comparison & Predictions</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        if yesterday_ohlc is not None:
            comparison_df = pd.DataFrame({
                'Yesterday': [float(x) for x in yesterday_ohlc],  # Convert to native floats
                'Today': [float(x) for x in today_ohlc],
                'Predicted (Next Day)': [float(x) for x in predicted_ohlc]
            }, index=ohlc_targets)
        else:
            comparison_df = pd.DataFrame({
                'Today': [float(x) for x in today_ohlc],
                'Predicted (Next Day)': [float(x) for x in predicted_ohlc]
            }, index=ohlc_targets)

        st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)

    with col2:
        # Ensemble close prediction
        st.markdown("### 🔮 Ensemble Prediction")
        st.metric("Next Day Close", f"₹{ensemble_close:.2f}",
                  f"{((ensemble_close / today_ohlc[3]) - 1) * 100:.2f}%")

        # Show model weights
        st.markdown("#### Model Weights:")
        for model, weight in weights.items():
            st.write(f"- {model}: {weight:.1%}")

    # Actual vs Predicted Close Plot
    actual_close = y_test['Close'].values.flatten()
    pred_close = y_ohlc_pred[:, 3].flatten()  # Close is at index 3

    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📉 Actual vs Predicted Close Price</h2>',
                unsafe_allow_html=True)

    plot_df = pd.DataFrame({
        'Actual': actual_close,
        'Predicted': pred_close
    }, index=y_test.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted Close'))
    fig.update_layout(
        title=f"Actual vs Predicted Close Prices (Backtest) - {stock_symbol}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Create table for last 7 working days
    last_7_days_data = plot_df.tail(7)
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">💰 Last 7 Working Days Performance</h2>',
                unsafe_allow_html=True)

    last_7_days_with_error = last_7_days_data.copy()
    last_7_days_with_error['Error (₹)'] = last_7_days_with_error['Predicted'] - last_7_days_with_error['Actual']
    last_7_days_with_error['Error (%)'] = (last_7_days_with_error['Error (₹)'] / last_7_days_with_error['Actual']) * 100

    st.dataframe(last_7_days_with_error.style.format({
        'Actual': '₹{:.2f}',
        'Predicted': '₹{:.2f}',
        'Error (₹)': '₹{:.2f}',
        'Error (%)': '{:.2f}%'
    }), use_container_width=True)

    # Backtesting Metrics
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📈 Backtesting Summary</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        mae = mean_absolute_error(actual_close, pred_close)
        mse = mean_squared_error(actual_close, pred_close)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_close, pred_close)

        backtest_metrics = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
            'Value': [mae, mse, rmse, r2]
        })

        st.dataframe(backtest_metrics.style.format({"Value": "{:.4f}"}), use_container_width=True)

    with col2:
        # User-Friendly Summary of Backtesting
        st.markdown("### What Do These Metrics Mean?")
        st.markdown(f"""
        - **MAE**: On average, predictions are ₹{mae:.2f} away from actual prices
        - **RMSE**: Error with higher penalty for large misses
        - **R² Score**: Model explains {r2 * 100:.2f}% of price movements
        - **Accuracy**: Approximately **{r2 * 100:.2f}%** accurate
        """)

    # Indicator Summary Table
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🧭 Technical Indicator Signals</h2>',
                unsafe_allow_html=True)

    # Get latest values
    latest = df.iloc[-1].copy()

    # Define required columns FIRST
    required_columns = ['MA5', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'Stoch_K']

    # Check for missing values
    if latest[required_columns].isna().any().item():
        st.error("❌ Missing values in technical indicators! Check your data.")
        st.stop()

    # Extract values safely
    ma5 = float(latest['MA5'])
    ma20 = float(latest['MA20'])
    rsi_value = float(latest['RSI'])
    macd_value = float(latest['MACD'])
    signal_value = float(latest['MACD_Signal'])
    stoch_k = float(latest['Stoch_K'])

    # Determine signals
    ma_signal = "Bullish" if ma5 > ma20 else "Bearish"
    rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
    macd_signal = "Bullish" if macd_value > signal_value else "Bearish"
    stoch_signal = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"

    # Now perform comparisons
    ma_signal = "Bullish" if ma5 > ma20 else "Bearish"
    rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
    macd_signal = "Bullish" if macd_value > signal_value else "Bearish"
    stoch_signal = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MA Signal", ma_signal, delta=f"MA5: {ma5:.2f}")
    with col2:
        st.metric("RSI Signal", rsi_signal, delta=f"RSI: {rsi_value:.2f}")
    with col3:
        st.metric("MACD Signal", macd_signal, delta=f"MACD: {macd_value:.4f}")
    with col4:
        st.metric("Stochastic Signal", stoch_signal, delta=f"K: {stoch_k:.2f}")

    # Overall trend
    st.markdown("### 📊 Overall Market Trend Prediction")
    try:
        trend_value = latest['Trend_Bollinger'].item()  # Convert to scalar
        trend_status = "Uptrend" if trend_value else "Downtrend"
    except KeyError:
        trend_status = "Neutral"
    st.write(f"The market for {stock_symbol} is predicted to be in an **{trend_status}**.")
    # Support and Resistance Analysis
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🎯 Support & Resistance Levels</h2>',
                unsafe_allow_html=True)

    try:
        # Get latest values as scalars
        latest = df.iloc[-1]
        latest_close = latest['Close'].item()
        latest_high = latest['High'].item()
        latest_low = latest['Low'].item()

        # Pivot Points
        pivot = (latest_high + latest_low + latest_close) / 3
        s1 = (2 * pivot) - latest_high
        s2 = pivot - (latest_high - latest_low)
        r1 = (2 * pivot) - latest_low
        r2 = pivot + (latest_high - latest_low)

        # Bollinger Bands
        upper_bb = latest['Bollinger_Upper'].item()
        lower_bb = latest['Bollinger_Lower'].item()
        ma20_value = latest['MA20'].item()

    except KeyError as e:
        st.error(f"Missing required column: {e}")
        st.stop()

    # Display in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Pivot Points")
        pp_df = pd.DataFrame({
            'Level': ['Resistance 2', 'Resistance 1', 'Pivot', 'Support 1', 'Support 2'],
            'Price': [r2, r1, pivot, s1, s2]
        })
        st.dataframe(pp_df.style.format({"Price": "₹{:.2f}"}), use_container_width=True)

    with col2:
        st.markdown("### Bollinger Bands")
        bb_df = pd.DataFrame({
            'Level': ['Upper Band', 'Middle Band (SMA20)', 'Lower Band'],
            'Price': [upper_bb, ma20_value, lower_bb]
        })
        st.dataframe(bb_df.style.format({"Price": "₹{:.2f}"}), use_container_width=True)

    # ------------------- Volume Profile Analysis -------------------


def analyze_volume_profile(df):
    """Analyze volume profile to identify support/resistance levels"""
    try:
        # Ensure we're working with a copy and numeric data
        df = df.copy()
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna(subset=['Volume'])

        # Create price bins
        price_min = df['Low'].min()
        price_max = df['High'].max()
        price_bins = np.linspace(price_min, price_max, 20)

        volume_profile = []
        for i in range(len(price_bins) - 1):
            lower_bound = price_bins[i]
            upper_bound = price_bins[i + 1]

            # Filter rows in this price range
            mask = (df['Close'] >= lower_bound) & (df['Close'] < upper_bound)
            bin_volume = df.loc[mask, 'Volume'].sum()

            # Store midpoint and volume
            midpoint = (lower_bound + upper_bound) / 2
            volume_profile.append((midpoint, bin_volume))

        # Create DataFrame with proper numeric types
        vp_df = pd.DataFrame(volume_profile, columns=['Price', 'Volume'])
        vp_df['Volume'] = pd.to_numeric(vp_df['Volume'], errors='coerce')

        # Get top 3 volume nodes
        high_volume_nodes = vp_df.nlargest(3, 'Volume')

        return vp_df, high_volume_nodes

    except Exception as e:
        raise ValueError(f"Error in volume profile analysis: {str(e)}")


# ------------------- Plotting and Display -------------------
def analyze_volume_profile(df):
    """Analyze volume profile to identify support/resistance levels"""
    try:
        df = df.copy()
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna(subset=['Close', 'Volume'])

        if len(df) == 0:
            raise ValueError("No valid data after cleaning")

        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
        volume_profile = []

        for i in range(len(price_bins) - 1):
            mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i + 1])
            bin_volume = df.loc[mask, 'Volume'].sum()
            volume_profile.append({
                'Price': (price_bins[i] + price_bins[i + 1]) / 2,
                'Volume': bin_volume
            })

        vp_df = pd.DataFrame(volume_profile)
        return vp_df, vp_df.nlargest(3, 'Volume')

    except Exception as e:
        raise ValueError(f"Volume profile error: {str(e)}")


# Create empty DataFrames to prevent undefined variable errors
vp_df = pd.DataFrame(columns=['Price', 'Volume'])
high_volume_nodes = pd.DataFrame(columns=['Price', 'Volume'])
# Show high volume nodes
st.dataframe(
    high_volume_nodes.style.format({
        'Price': '₹{:.2f}',
        'Volume': '{:,}'
    }),
    use_container_width=True
)

# ====================== Market Sentiment Tab ======================
with tab2:
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📰 Market Sentiment Analysis</h2>',
                unsafe_allow_html=True)


    # News API for market sentiment
    @st.cache_data(ttl=3600)
    def get_news_sentiment(query):
        try:
            # Note: In a production app, you would use a real news API with your API key
            # This is a placeholder implementation
            st.info("News sentiment analysis would be implemented with a real API in production")

            # Simulate some sentiment data
            return {
                'positive': 0.65,
                'negative': 0.25,
                'neutral': 0.10,
                'sample_headlines': [
                    "Company announces record profits",
                    "New product launch exceeds expectations",
                    "Market analysts raise price target",
                    "Concerns about supply chain issues",
                    "Competitor enters market"
                ]
            }
        except Exception as e:
            st.error(f"Error fetching news sentiment: {e}")
            return None


    # Twitter sentiment (placeholder)
    @st.cache_data(ttl=3600)
    def get_twitter_sentiment(query):
        try:
            # Note: In a production app, you would use Twitter API
            st.info("Twitter sentiment analysis would be implemented with a real API in production")

            # Simulate some sentiment data
            return {
                'positive': 0.55,
                'negative': 0.30,
                'neutral': 0.15,
                'sample_tweets': [
                    "Just bought more $BEL - looking strong!",
                    "Why is $BEL dropping today?",
                    "Holding $BEL for long term growth",
                    "Not a good day for $BEL holders",
                    "Technical indicators look bullish for $BEL"
                ]
            }
        except Exception as e:
            st.error(f"Error fetching Twitter sentiment: {e}")
            return None


    # Analyze sentiment
    query = stock_symbol.split('.')[0]  # Remove .NS suffix if present
    with st.spinner("Analyzing market sentiment..."):
        news_sentiment = get_news_sentiment(query)
        twitter_sentiment = get_twitter_sentiment(query)

    if news_sentiment and twitter_sentiment:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📰 News Sentiment")
            fig_news = go.Figure(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[news_sentiment['positive'], news_sentiment['negative'], news_sentiment['neutral']],
                hole=0.4
            ))
            fig_news.update_layout(title_text="News Sentiment Distribution")
            st.plotly_chart(fig_news, use_container_width=True)

            st.markdown("**Sample Headlines:**")
            for headline in news_sentiment['sample_headlines']:
                st.write(f"- {headline}")

        with col2:
            st.markdown("### 🐦 Twitter Sentiment")
            fig_twitter = go.Figure(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[twitter_sentiment['positive'], twitter_sentiment['negative'], twitter_sentiment['neutral']],
                hole=0.4
            ))
            fig_twitter.update_layout(title_text="Twitter Sentiment Distribution")
            st.plotly_chart(fig_twitter, use_container_width=True)

            st.markdown("**Sample Tweets:**")
            for tweet in twitter_sentiment['sample_tweets']:
                st.write(f"- {tweet}")

    # Combined sentiment score
    if news_sentiment and twitter_sentiment:
        combined_score = (news_sentiment['positive'] * 0.6 + twitter_sentiment['positive'] * 0.4) * 100
        st.markdown("### 🎯 Overall Sentiment Score")
        st.metric("Bullish Sentiment", f"{combined_score:.1f}/100",
                  delta=f"{'↑ Bullish' if combined_score > 50 else '↓ Bearish'}")

        # Sentiment interpretation
        if combined_score > 70:
            st.success("Strong positive sentiment detected. Market participants are generally optimistic.")
        elif combined_score > 50:
            st.info("Moderately positive sentiment detected. Cautious optimism prevails.")
        elif combined_score > 30:
            st.warning("Negative sentiment detected. Market participants are concerned.")
        else:
            st.error("Strong negative sentiment detected. Significant pessimism in the market.")

# ====================== Advanced Indicators Tab ======================
with tab3:
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📊 Advanced Technical Indicators</h2>',
                unsafe_allow_html=True)

    # Select indicators to display
    indicator_options = [
        'MACD', 'RSI', 'Bollinger Bands', 'Stochastic Oscillator',
        'ATR', 'OBV', 'Ichimoku Cloud', 'All Indicators'
    ]
    selected_indicators = st.multiselect(
        "Select indicators to visualize:",
        indicator_options,
        default=['MACD', 'RSI']
    )

    # Create subplots based on selected indicators
    if selected_indicators:
        num_plots = len(selected_indicators) + 1  # +1 for price chart
        fig_indicators = make_subplots(
            rows=num_plots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Price Chart'] + [f"{ind} Indicator" for ind in selected_indicators]
        )

        # Price chart
        fig_indicators.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add moving averages to price chart
        fig_indicators.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA20'],
                line=dict(color='blue', width=1),
                name='20-day MA'
            ),
            row=1, col=1
        )

        fig_indicators.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA50'],
                line=dict(color='orange', width=1),
                name='50-day MA'
            ),
            row=1, col=1
        )

        # Add selected indicators
        for i, indicator in enumerate(selected_indicators, start=2):
            if indicator == 'MACD':
                fig_indicators.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        line=dict(color='blue', width=1),
                        name='MACD'
                    ),
                    row=i, col=1
                )

                fig_indicators.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD_Signal'],
                        line=dict(color='red', width=1),
                        name='Signal Line'
                    ),
                    row=i, col=1
                )

                # Add histogram
                fig_indicators.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['MACD_Diff'],
                        marker_color=np.where(df['MACD_Diff'] > 0, 'green', 'red'),
                        name='MACD Histogram'
                    ),
                    row=i, col=1
                )

            elif indicator == 'RSI':
                fig_indicators.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        line=dict(color='purple', width=1),
                        name='RSI'
                    ),
                    row=i, col=1
                )

                # Add overbought/oversold lines
                fig_indicators.add_hline(
                    y=70, line_dash="dash", line_color="red",
                    annotation_text="Overbought",
                    annotation_position="bottom right",
                    row=i, col=1
                )

                fig_indicators.add_hline(
                    y=30, line_dash="dash", line_color="green",
                    annotation_text="Oversold",
                    annotation_position="bottom right",
                    row=i, col=1
                )

            # Add similar blocks for other indicators...

        # Update layout
        fig_indicators.update_layout(
            height=200 * num_plots,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        # Disable zoom on all but the bottom plot
        for i in range(1, num_plots):
            fig_indicators.update_xaxes(showticklabels=False, row=i, col=1)

        st.plotly_chart(fig_indicators, use_container_width=True)

    # Indicator interpretation guide
    with st.expander("📖 Indicator Interpretation Guide"):
        st.markdown("""
            ### Technical Indicator Guide

            **MACD (Moving Average Convergence Divergence)**
                - Bullish when MACD crosses above signal line
                - Bearish when MACD crosses below signal line
                - Histogram shows strength of trend

             **RSI (Relative Strength Index)**
                - Above 70: Overbought (potential reversal)
                - Below 30: Oversold (potential reversal)
                - Divergence between price and RSI can signal reversals

            **Bollinger Bands**
                - Price near upper band: Overbought
                - Price near lower band: Oversold
                - Band width indicates volatility

            **Stochastic Oscillator**
                - %K above %D: Bullish momentum
                - %K below %D: Bearish momentum
                - Above 80: Overbought, Below 20: Oversold

            **ATR (Average True Range)**
                - Measures market volatility
                - Higher values indicate more volatility

            **OBV (On Balance Volume)**
                - Rising OBV confirms price uptrend
                - Falling OBV confirms price downtrend
                - Divergences can signal reversals

            **Ichimoku Cloud**
                - Price above cloud: Bullish trend
                - Price below cloud: Bearish trend
                - Cloud color change signals trend change
                """)

# ====================== Risk Assessment Tab ======================
with tab4:
    st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">⚠️ Risk Assessment & Portfolio Analysis</h2>',
                unsafe_allow_html=True)

    # Add this line to define beta
    beta = 1.0  # Placeholder value for demonstration

    # Calculate risk metrics with proper scalar conversion
    returns = df['Close'].pct_change().dropna()


    def to_scalar(value):
        """Convert pandas Series/array to scalar float"""
        if hasattr(value, 'iloc'):
            return float(value.iloc[0])
        elif hasattr(value, 'item'):
            return float(value.item())
        return float(value)


    # Calculate and convert metrics
    volatility = to_scalar(returns.std() * np.sqrt(252))
    max_drawdown = to_scalar((df['Close'] / df['Close'].cummax() - 1).min())
    sharpe_ratio = to_scalar(returns.mean() / returns.std() * np.sqrt(252))

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annualized Volatility", f"{volatility * 100:.2f}%")
    with col2:
        st.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Beta (Systematic Risk)", f"{beta:.2f}")

    # Value at Risk calculation
    confidence_level = st.slider("Select confidence level:", 90, 99, 95)

    # Historical VaR
    historical_var = to_scalar(np.percentile(returns, 100 - confidence_level) * 100)

    # Parametric VaR
    parametric_var = to_scalar(
        (returns.mean() - (1.645 * returns.std())) * 100
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            f"Historical VaR ({confidence_level}% confidence)",
            f"{historical_var:.2f}%"
        )
    with col2:
        st.metric(
            f"Parametric VaR ({confidence_level}% confidence)",
            f"{parametric_var:.2f}%"
        )

    # Monte Carlo simulation (placeholder)
    if st.checkbox("Show Monte Carlo Simulation (Placeholder)"):
        st.info("In a full implementation, this would show Monte Carlo simulated price paths")

        # Placeholder plot
        mc_fig = go.Figure()
        for _ in range(10):  # Simulate 10 paths
            simulated = df['Close'].iloc[-1] * np.exp(np.cumsum(np.random.normal(
                returns.mean(),
                returns.std(),
                30
            )))
            mc_fig.add_trace(
                go.Scatter(
                    x=pd.date_range(df.index[-1], periods=30),
                    y=simulated,
                    line=dict(width=1),
                    showlegend=False
                )
            )

        mc_fig.add_trace(
            go.Scatter(
                x=df.index[-30:],
                y=df['Close'].iloc[-30:],
                line=dict(color='black', width=2),
                name='Actual Price'
            )
        )

        mc_fig.update_layout(
            title="Monte Carlo Simulation (30-day projection)",
            yaxis_title="Price"
        )
        st.plotly_chart(mc_fig, use_container_width=True)

# ====================== Footer ======================
st.markdown("---")
st.markdown("""
            <div style="text-align: center; color: #666; font-size: 14px;">
                <p>Disclaimer: This app is for educational purposes only. Stock market investments are subject to risks.</p>
                <p>The predictions and analysis should not be considered as financial advice.</p>
                <p>Always conduct your own research before making investment decisions.</p>
            </div>
        """, unsafe_allow_html=True)

# Add refresh button
if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
