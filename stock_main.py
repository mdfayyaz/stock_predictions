import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="NSE Stock Price Predictor", layout="wide")

# App title and description
st.title("NSE Stock Price Predictor")
st.markdown("""
This application predicts the next day's Open, High, Low and Close prices for NSE stocks using 
advanced machine learning algorithms and technical indicators.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Function to get NSE stock symbols
@st.cache_data
def get_nse_stocks():
    # In a real scenario, you would fetch from NSE API or a reliable source
    # For demonstration, using a few well-known NSE stocks with NSE suffix
    nse_stocks = {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'SBIN.NS': 'State Bank of India',
        'BAJFINANCE.NS': 'Bajaj Finance',
        'ICICIBANK.NS': 'ICICI Bank',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank',
        'AXISBANK.NS': 'Axis Bank',
        'WIPRO.NS': 'Wipro',
        'HCLTECH.NS': 'HCL Technologies',
        'TATAMOTORS.NS': 'Tata Motors',
        'SUNPHARMA.NS': 'Sun Pharmaceutical',
        'ASIANPAINT.NS': 'Asian Paints',
        'NESTLEIND.NS': 'Nestle India',
        'MARUTI.NS': 'Maruti Suzuki India',
        'NTPC.NS': 'NTPC',
        'POWERGRID.NS': 'Power Grid Corporation'
    }
    return nse_stocks

nse_stocks = get_nse_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", list(nse_stocks.keys()), format_func=lambda x: f"{nse_stocks[x]} ({x})")

# Date range for historical data
end_date = datetime.now().date()
start_date = end_date - timedelta(days=730)  # Two years of data
start_date = st.sidebar.date_input("Start Date", start_date)
end_date = st.sidebar.date_input("End Date", end_date)

if start_date >= end_date:
    st.error("End date should be greater than start date")

# Model selection
model_options = ["XGBoost", "Random Forest", "Gradient Boosting", "Ensemble (All Models)"]
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Training window
training_window = st.sidebar.slider("Training Window (Days)", min_value=30, max_value=500, value=365)

# Features to include
st.sidebar.subheader("Features to Include")
use_price_features = st.sidebar.checkbox("Price Features", value=True)
use_momentum_indicators = st.sidebar.checkbox("Momentum Indicators", value=True)
use_trend_indicators = st.sidebar.checkbox("Trend Indicators", value=True)
use_volatility_indicators = st.sidebar.checkbox("Volatility Indicators", value=True)
use_volume_indicators = st.sidebar.checkbox("Volume Indicators", value=True)
use_sentiment = st.sidebar.checkbox("Market Sentiment", value=False)  # For future implementation

# Advanced options
st.sidebar.subheader("Advanced Options")
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=30, value=20) / 100
n_estimators = st.sidebar.slider("Number of Estimators", min_value=50, max_value=500, value=200)
max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=15, value=10)

# Function to download stock data
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the stock symbol and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to add technical indicators
def add_technical_indicators(df):
    df_copy = df.copy()
    
    if use_momentum_indicators:
        # RSI
        rsi = RSIIndicator(close=df_copy['Close'], window=14)
        df_copy['RSI'] = rsi.rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'], window=14, smooth_window=3)
        df_copy['Stoch_K'] = stoch.stoch()
        df_copy['Stoch_D'] = stoch.stoch_signal()
    
    if use_trend_indicators:
        # MACD
        macd = MACD(close=df_copy['Close'])
        df_copy['MACD'] = macd.macd()
        df_copy['MACD_Signal'] = macd.macd_signal()
        df_copy['MACD_Diff'] = macd.macd_diff()
        
        # ADX
        adx = ADXIndicator(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'])
        df_copy['ADX'] = adx.adx()
        
        # SMAs
        for window in [5, 20, 50, 200]:
            sma = SMAIndicator(close=df_copy['Close'], window=window)
            df_copy[f'SMA_{window}'] = sma.sma_indicator()
        
        # EMAs
        for window in [5, 20, 50, 200]:
            ema = EMAIndicator(close=df_copy['Close'], window=window)
            df_copy[f'EMA_{window}'] = ema.ema_indicator()
    
    if use_volatility_indicators:
        # Bollinger Bands
        bb = BollingerBands(close=df_copy['Close'])
        df_copy['BB_High'] = bb.bollinger_hband()
        df_copy['BB_Low'] = bb.bollinger_lband()
        df_copy['BB_Mid'] = bb.bollinger_mavg()
        df_copy['BB_Width'] = df_copy['BB_High'] - df_copy['BB_Low']
        df_copy['BB_%B'] = (df_copy['Close'] - df_copy['BB_Low']) / (df_copy['BB_High'] - df_copy['BB_Low'])
        
        # ATR
        atr = AverageTrueRange(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'])
        df_copy['ATR'] = atr.average_true_range()
    
    if use_volume_indicators and 'Volume' in df_copy.columns:
        # MFI
        try:
            mfi = MFIIndicator(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'], volume=df_copy['Volume'])
            df_copy['MFI'] = mfi.money_flow_index()
        except:
            pass
        
        # OBV
        try:
            obv = OnBalanceVolumeIndicator(close=df_copy['Close'], volume=df_copy['Volume'])
            df_copy['OBV'] = obv.on_balance_volume()
        except:
            pass
    
    if use_price_features:
        # Price Changes
        df_copy['Price_Change'] = df_copy['Close'].pct_change()
        df_copy['Price_Change_1d'] = df_copy['Close'].pct_change(1)
        df_copy['Price_Change_5d'] = df_copy['Close'].pct_change(5)
        df_copy['Price_Change_10d'] = df_copy['Close'].pct_change(10)
        df_copy['Price_Change_20d'] = df_copy['Close'].pct_change(20)
        
        # Volatility
        df_copy['Volatility_5d'] = df_copy['Close'].pct_change().rolling(window=5).std()
        df_copy['Volatility_10d'] = df_copy['Close'].pct_change().rolling(window=10).std()
        df_copy['Volatility_20d'] = df_copy['Close'].pct_change().rolling(window=20).std()
        
        # High-Low Range
        df_copy['HL_Range'] = (df_copy['High'] - df_copy['Low']) / df_copy['Close']
        df_copy['HL_Range_MA5'] = df_copy['HL_Range'].rolling(window=5).mean()
        
        # Open-Close Range
        df_copy['OC_Range'] = abs(df_copy['Open'] - df_copy['Close']) / df_copy['Close']
        
        # Gap Up/Down
        df_copy['Gap'] = (df_copy['Open'] - df_copy['Close'].shift(1)) / df_copy['Close'].shift(1)
    
    # Lag features for OHLC
    for i in range(1, 6):
        df_copy[f'Open_Lag{i}'] = df_copy['Open'].shift(i)
        df_copy[f'High_Lag{i}'] = df_copy['High'].shift(i)
        df_copy[f'Low_Lag{i}'] = df_copy['Low'].shift(i)
        df_copy[f'Close_Lag{i}'] = df_copy['Close'].shift(i)
    
    # Drop NaN values
    df_copy = df_copy.dropna()
    
    return df_copy

# Function to prepare data for modeling
def prepare_data(df, target_col, forecast_horizon=1):
    # Create target variables (next day's values)
    df[f'Next_Day_{target_col}'] = df[target_col].shift(-forecast_horizon)
    
    # Drop rows with NaN in the target variable
    df = df.dropna(subset=[f'Next_Day_{target_col}'])
    
    # Separate features and target
    X = df.drop(['Next_Day_Open', 'Next_Day_High', 'Next_Day_Low', 'Next_Day_Close'], errors='ignore')
    X = X.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    y = df[f'Next_Day_{target_col}']
    
    return X, y

# Function to train models
def train_model(X, y, model_type):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_type == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
    elif model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, rmse, mae, r2, X_test, y_test, y_pred

# Function for time series cross-validation backtesting
def backtest_model(df, target_col, model_type, n_splits=5):
    # Prepare data
    df_prepared = add_technical_indicators(df)
    X, y = prepare_data(df_prepared, target_col)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Metrics storage
    metrics = {
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    # Loop through splits
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
        elif model_type == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
    
    # Return average metrics
    avg_metrics = {
        'rmse': np.mean(metrics['rmse']),
        'mae': np.mean(metrics['mae']),
        'r2': np.mean(metrics['r2'])
    }
    
    return avg_metrics

# Function to make next day prediction
def predict_next_day(df, target_col, model_type):
    # Prepare data
    df_prepared = add_technical_indicators(df)
    X, y = prepare_data(df_prepared, target_col)
    
    # Train model on all data
    model, scaler, _, _, _, _, _, _ = train_model(X, y, model_type)
    
    # Prepare latest data for prediction
    latest_data = df_prepared.iloc[-1:].drop(['Next_Day_Open', 'Next_Day_High', 'Next_Day_Low', 'Next_Day_Close'], errors='ignore')
    latest_data = latest_data.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    
    # Scale and predict
    latest_data_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_data_scaled)[0]
    
    return prediction

# Function to run ensemble model
def ensemble_prediction(df, target_col):
    models = ["XGBoost", "Random Forest", "Gradient Boosting"]
    predictions = []
    
    for model_type in models:
        pred = predict_next_day(df, target_col, model_type)
        predictions.append(pred)
    
    # Return average prediction
    return np.mean(predictions)

# Function to display feature importance
def display_feature_importance(X, model, model_type):
    if model_type == "XGBoost":
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), 
                                  columns=['Value', 'Feature'])
    elif model_type in ["Random Forest", "Gradient Boosting"]:
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), 
                                  columns=['Value', 'Feature'])
    else:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
    plt.title(f'Top 20 Feature Importances - {model_type}')
    plt.tight_layout()
    
    return fig

# Function to display additional insights
def display_additional_insights(df):
    """Function to display additional insights based on the dataframe"""
    st.markdown("---")
    st.subheader("Additional Insights")

    # Display tips based on technical indicators
    try:
        latest_data = df.iloc[-1]
        prev_data = df.iloc[-2]
        
        # Calculate some basic signals
        current_close = latest_data['Close']
        prev_close = prev_data['Close']
        
        # SMA calculations
        short_sma = df['Close'].rolling(window=20).mean().iloc[-1]
        long_sma = df['Close'].rolling(window=50).mean().iloc[-1]
        
        insights = []
        
        # Price movement
        if current_close > prev_close:
            insights.append("• Stock closed higher yesterday, showing positive momentum.")
        else:
            insights.append("• Stock closed lower yesterday, showing negative momentum.")
        
        # SMA crossover
        if short_sma > long_sma and df['Close'].rolling(window=20).mean().iloc[-2] <= df['Close'].rolling(window=50).mean().iloc[-2]:
            insights.append("• Golden Cross detected (short-term SMA crossing above long-term SMA), potentially bullish signal.")
        elif short_sma < long_sma and df['Close'].rolling(window=20).mean().iloc[-2] >= df['Close'].rolling(window=50).mean().iloc[-2]:
            insights.append("• Death Cross detected (short-term SMA crossing below long-term SMA), potentially bearish signal.")
        
        # Price relative to SMAs
        if current_close > short_sma and current_close > long_sma:
            insights.append("• Price is above both short and long-term moving averages, indicating bullish trend.")
        elif current_close < short_sma and current_close < long_sma:
            insights.append("• Price is below both short and long-term moving averages, indicating bearish trend.")
        
        # Volume analysis
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            if latest_data['Volume'] > avg_volume * 1.5:
                insights.append("• Trading volume is significantly higher than average, indicating strong interest.")
            elif latest_data['Volume'] < avg_volume * 0.5:
                insights.append("• Trading volume is significantly lower than average, indicating weak interest.")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.markdown("No additional insights available.")
    except Exception as e:
        st.error(f"Error generating insights: {e}")

# Main app logic
if st.button("Run Analysis"):
    # Show spinner while processing
    with st.spinner("Fetching and analyzing data..."):
        # Get stock data
        df = get_stock_data(selected_stock, start_date, end_date)
        
        if df is not None and not df.empty:
            # Display basic stock info
            st.subheader(f"Stock Data: {nse_stocks[selected_stock]} ({selected_stock})")
            
            # Show stock price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            ))
            fig.update_layout(
                title=f"{nse_stocks[selected_stock]} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Technical Analysis", "Predictions", "Feature Importance"])
            
            with tab1:
                st.subheader("Model Performance Metrics")
                
                # Use the most recent data for training based on window
                recent_df = df.iloc[-training_window:]
                
                # Prepare data with indicators
                df_indicators = add_technical_indicators(recent_df)
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Train models and show metrics for each target
                with col1:
                    st.markdown("### Open Price")
                    if selected_model == "Ensemble (All Models)":
                        metrics_open = {}
                        for model in ["XGBoost", "Random Forest", "Gradient Boosting"]:
                            metrics_open[model] = backtest_model(recent_df, 'Open', model)
                        
                        # Average the metrics
                        avg_rmse = np.mean([metrics_open[m]['rmse'] for m in metrics_open])
                        avg_mae = np.mean([metrics_open[m]['mae'] for m in metrics_open])
                        avg_r2 = np.mean([metrics_open[m]['r2'] for m in metrics_open])
                        
                        st.metric("RMSE", f"₹{avg_rmse:.2f}")
                        st.metric("MAE", f"₹{avg_mae:.2f}")
                        st.metric("R² Score", f"{avg_r2:.4f}")
                    else:
                        metrics_open = backtest_model(recent_df, 'Open', selected_model)
                        st.metric("RMSE", f"₹{metrics_open['rmse']:.2f}")
                        st.metric("MAE", f"₹{metrics_open['mae']:.2f}")
                        st.metric("R² Score", f"{metrics_open['r2']:.4f}")
                
                with col2:
                    st.markdown("### High Price")
                    if selected_model == "Ensemble (All Models)":
                        metrics_high = {}
                        for model in ["XGBoost", "Random Forest", "Gradient Boosting"]:
                            metrics_high[model] = backtest_model(recent_df, 'High', model)
                        
                        # Average the metrics
                        avg_rmse = np.mean([metrics_high[m]['rmse'] for m in metrics_high])
                        avg_mae = np.mean([metrics_high[m]['mae'] for m in metrics_high])
                        avg_r2 = np.mean([metrics_high[m]['r2'] for m in metrics_high])
                        
                        st.metric("RMSE", f"₹{avg_rmse:.2f}")
                        st.metric("MAE", f"₹{avg_mae:.2f}")
                        st.metric("R² Score", f"{avg_r2:.4f}")
                    else:
                        metrics_high = backtest_model(recent_df, 'High', selected_model)
                        st.metric("RMSE", f"₹{metrics_high['rmse']:.2f}")
                        st.metric("MAE", f"₹{metrics_high['mae']:.2f}")
                        st.metric("R² Score", f"{metrics_high['r2']:.4f}")
                
                with col3:
                    st.markdown("### Low Price")
                    if selected_model == "Ensemble (All Models)":
                        metrics_low = {}
                        for model in ["XGBoost", "Random Forest", "Gradient Boosting"]:
                            metrics_low[model] = backtest_model(recent_df, 'Low', model)
                        
                        # Average the metrics
                        avg_rmse = np.mean([metrics_low[m]['rmse'] for m in metrics_low])
                        avg_mae = np.mean([metrics_low[m]['mae'] for m in metrics_low])
                        avg_r2 = np.mean([metrics_low[m]['r2'] for m in metrics_low])
                        
                        st.metric("RMSE", f"₹{avg_rmse:.2f}")
                        st.metric("MAE", f"₹{avg_mae:.2f}")
                        st.metric("R² Score", f"{avg_r2:.4f}")
                    else:
                        metrics_low = backtest_model(recent_df, 'Low', selected_model)
                        st.metric("RMSE", f"₹{metrics_low['rmse']:.2f}")
                        st.metric("MAE", f"₹{metrics_low['mae']:.2f}")
                        st.metric("R² Score", f"{metrics_low['r2']:.4f}")
                
                with col4:
                    st.markdown("### Close Price")
                    if selected_model == "Ensemble (All Models)":
                        metrics_close = {}
                        for model in ["XGBoost", "Random Forest", "Gradient Boosting"]:
                            metrics_close[model] = backtest_model(recent_df, 'Close', model)
                        
                        # Average the metrics
                        avg_rmse = np.mean([metrics_close[m]['rmse'] for m in metrics_close])
                        avg_mae = np.mean([metrics_close[m]['mae'] for m in metrics_close])
                        avg_r2 = np.mean([metrics_close[m]['r2'] for m in metrics_close])
                        
                        st.metric("RMSE", f"₹{avg_rmse:.2f}")
                        st.metric("MAE", f"₹{avg_mae:.2f}")
                        st.metric("R² Score", f"{avg_r2:.4f}")
                    else:
                        metrics_close = backtest_model(recent_df, 'Close', selected_model)
                        st.metric("RMSE", f"₹{metrics_close['rmse']:.2f}")
                        st.metric("MAE", f"₹{metrics_close['mae']:.2f}")
                        st.metric("R² Score", f"{metrics_close['r2']:.4f}")
            
            with tab2:
                st.subheader("Technical Analysis")
                
                # Display technical indicators
                tech_df = add_technical_indicators(df.iloc[-60:])  # Last 60 days for visualization
                
                # Technical Analysis Plots
                st.markdown("### Price with Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=tech_df.index,
                    open=tech_df['Open'],
                    high=tech_df['High'],
                    low=tech_df['Low'],
                    close=tech_df['Close'],
                    name='Price'
                ))
                
                if 'SMA_20' in tech_df.columns:
                    fig.add_trace(go.Scatter(
                        x=tech_df.index,
                        y=tech_df['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='blue')
                    )
                
                if 'SMA_50' in tech_df.columns:
                    fig.add_trace(go.Scatter(
                        x=tech_df.index,
                        y=tech_df['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='orange')
                    )
                
                if 'EMA_20' in tech_df.columns:
                    fig.add_trace(go.Scatter(
                        x=tech_df.index,
                        y=tech_df['EMA_20'],
                        mode='lines',
                        name='EMA 20',
                        line=dict(color='green', dash='dash')
                    )
                
                fig.update_layout(
                    title="Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Plot
                if 'RSI' in tech_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=tech_df.index,
                        y=tech_df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    )
                    
                    # Add horizontal lines at 30 and 70
                    fig.add_shape(
                        type="line",
                        x0=tech_df.index[0], y0=30,
                        x1=tech_df.index[-1], y1=30,
                        line=dict(color="red", dash="dash")
                    )
                    fig.add_shape(
                        type="line",
                        x0=tech_df.index[0], y0=70,
                        x1=tech_df.index[-1], y1=70,
                        line=dict(color="red", dash="dash")
                    )
                    fig.update_layout(
                        title="Relative Strength Index (RSI)",
                        yaxis=dict(range=[0, 100]),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Next Day Price Predictions")
                try:
                    recent_df = df.iloc[-training_window:]

                    if selected_model == "Ensemble (All Models)":
                        next_open = ensemble_prediction(recent_df, 'Open')
                        next_high = ensemble_prediction(recent_df, 'High')
                        next_low = ensemble_prediction(recent_df, 'Low')
                        next_close = ensemble_prediction(recent_df, 'Close')
                    else:
                        next_open = predict_next_day(recent_df, 'Open', selected_model)
                        next_high = predict_next_day(recent_df, 'High', selected_model)
                        next_low = predict_next_day(recent_df, 'Low', selected_model)
                        next_close = predict_next_day(recent_df, 'Close', selected_model)

                    next_trading_day = df.index[-1] + pd.Timedelta(days=1)
                    if next_trading_day.weekday() >= 5:  # Handle weekends
                        next_trading_day += pd.Timedelta(days=7 - next_trading_day.weekday())

                    st.markdown(f"**Predictions for {next_trading_day.strftime('%A, %B %d, %Y')}**")
                    current_close = df['Close'].iloc[-1]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Open", f"₹{next_open:.2f}", 
                                  f"{((next_open - current_close)/current_close*100:.2f}%")
                    with col2:
                        st.metric("High", f"₹{next_high:.2f}", 
                                  f"{((next_high - current_close)/current_close*100:.2f}%")
                    with col3:
                        st.metric("Low", f"₹{next_low:.2f}", 
                                  f"{((next_low - current_close)/current_close*100:.2f}%")
                    with col4:
                        st.metric("Close", f"₹{next_close:.2f}", 
                                  f"{((next_close - current_close)/current_close*100:.2f}%")

                except Exception as e:
                    st.error(f"Error generating predictions: {e}")

            with tab4:
                st.subheader("Feature Importance Analysis")
                try:
                    df_prepared = add_technical_indicators(recent_df)
                    targets = ['Open', 'High', 'Low', 'Close']

                    for target in targets:
                        st.markdown(f"### {target} Price Features")
                        if selected_model != "Ensemble (All Models)":
                            X, y = prepare_data(df_prepared, target)
                            model, _, _, _, _, _, _, _ = train_model(X, y, selected_model)
                            fig = display_feature_importance(X, model, selected_model)
                            st.pyplot(fig)
                        else:
                            st.info("Feature importance not available for Ensemble model")
                            break
                except Exception as e:
                    st.error(f"Error generating feature importance: {e}")

            # Display additional insights
            display_additional_insights(df)

            # Footer
            st.markdown("---")
            st.markdown("""
            **Disclaimer**: This tool is for educational purposes only. 
            Past performance is not indicative of future results.
            """)
