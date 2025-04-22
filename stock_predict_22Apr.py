import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import pytz

# Configure Streamlit page
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Advanced Stock Price Predictor")

# Initialize NewsAPI from Streamlit secrets
try:
    newsapi = NewsApiClient(api_key=st.secrets["newsapi"]["api_key"])
except:
    st.error("NewsAPI key not found in secrets. Please configure secrets.toml")
    newsapi = None


# --- Technical Indicators ---
def calculate_technical_indicators(df):
    df = df.copy()
    close = df['Close'].copy()
    volume = df['Volume'].copy()

    # Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volume Indicators
    vol_ma = volume.rolling(window=20).mean()
    df['Volume_MA'] = vol_ma
    df['Volume_Ratio'] = volume / vol_ma

    # Momentum Indicators
    df['Momentum_5D'] = close.pct_change(5)
    df['Volatility'] = close.rolling(window=20).std()

    # Bollinger Bands
    middle_band = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    df['Middle_Band'] = middle_band
    df['Upper_Band'] = middle_band + (2 * rolling_std)
    df['Lower_Band'] = middle_band - (2 * rolling_std)

    df.dropna(inplace=True)
    return df


# --- Risk Assessment ---
def calculate_risk(predicted_change):
    if abs(predicted_change) > 0.07:
        return "🔴 High Risk"
    elif abs(predicted_change) > 0.03:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


# --- Get Latest Stock Price ---
def get_latest_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        # Determine timezone based on ticker
        tz = pytz.timezone('Asia/Kolkata') if '.NS' in ticker else pytz.timezone('America/New_York')

        # Get today's data
        today_data = stock.history(period='1d')
        if not today_data.empty:
            latest_price = today_data['Close'].iloc[-1]
            latest_time = today_data.index[-1]
            if latest_time.tzinfo is None:
                latest_time = pytz.utc.localize(latest_time)
            latest_time = latest_time.astimezone(tz)
            return latest_price, latest_time

        # Fallback to previous close
        prev_close = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', None))
        hist_data = stock.history(period='5d')
        if not hist_data.empty:
            prev_time = hist_data.index[-1]
            if prev_time.tzinfo is None:
                prev_time = pytz.utc.localize(prev_time)
            prev_time = prev_time.astimezone(tz)
            return prev_close, prev_time

        return prev_close, None
    except Exception as e:
        st.error(f"Error fetching price: {str(e)}")
        return None, None


# --- Get Stock News ---
def get_stock_news(ticker, days=30):
    try:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker.split('.')[0])

        # Try NewsAPI first if available
        news_items = []
        if newsapi:
            response = newsapi.get_everything(
                q=f"{company_name} OR {ticker.split('.')[0]}",
                from_param=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=30
            )
            for article in response.get('articles', []):
                news_items.append({
                    'date': article['publishedAt'],
                    'title': article['title'],
                    'publisher': article['source']['name'],
                    'description': article['description'],
                    'url': article['url'],
                    'image': article['urlToImage']
                })

        # Fallback to Yahoo Finance
        if not news_items:
            yf_news = stock.news
            if yf_news:
                for item in yf_news:
                    pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                    news_items.append({
                        'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'title': item['title'],
                        'publisher': item['publisher'],
                        'url': item['link'],
                        'description': '',
                        'image': ''
                    })

        return sorted(news_items, key=lambda x: x['date'], reverse=True)[:30]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []


# --- Update Features with Latest Price ---
def update_features_with_latest_price(features, latest_price):
    features = features.copy()
    features['Close'] = latest_price

    # Recalculate indicators that depend on Close price
    if 'SMA_20' in features:
        features['SMA_20'] = features['Close'].rolling(window=20).mean().iloc[-1]
    if 'EMA_20' in features:
        features['EMA_20'] = features['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    if 'RSI' in features:
        delta = features['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14).mean().iloc[-1]
        rs = avg_gain / avg_loss
        features['RSI'] = 100 - (100 / (1 + rs))

    return features


# --- Main App ---
def main():
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Stock Ticker (e.g., TCS.NS)", "TCS.NS")
    days_back = st.sidebar.slider("Backtest Period (days)", 30, 365, 90)
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost"])

    # Load data
    @st.cache_data
    def load_data(ticker):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)
        return yf.download(ticker, start=start_date, end=end_date)

    data = load_data(ticker)
    if data.empty:
        st.error("Failed to load data. Check ticker symbol.")
        return

    # Get and display latest price
    latest_price, price_time = get_latest_price(ticker)
    if latest_price is None:
        st.error("Could not fetch latest price")
        return

    price_str = f"₹{latest_price:.2f}" if '.NS' in ticker else f"${latest_price:.2f}"
    if price_time:
        st.sidebar.metric("Latest Price", price_str, f"As of {price_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.sidebar.metric("Latest Price", price_str, "Previous Close")

    # Calculate indicators
    data = calculate_technical_indicators(data)
    data['Next_Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    # Train-test split
    split_idx = int(len(data) * 0.8)
    X = data.drop(['Next_Close'], axis=1)
    y = data['Next_Close']
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model training
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor()
    model.fit(X_train, y_train)

    # --- Prediction Display ---
    st.header("📊 Current Prediction")

    # Get the latest features and update with current price
    latest_features = X.iloc[[-1]].copy()
    updated_features = update_features_with_latest_price(latest_features, latest_price)

    # Make prediction
    predicted_price = float(model.predict(updated_features)[0])
    change_pct = float((predicted_price - latest_price) / latest_price * 100)
    currency = '₹' if '.NS' in ticker else '$'

    cols = st.columns(3)
    cols[0].metric("Current Close", f"{currency}{latest_price:.2f}",
                   f"As of {price_time.strftime('%Y-%m-%d %H:%M')}" if price_time else "Latest Available")
    cols[1].metric("Predicted Close", f"{currency}{predicted_price:.2f}", f"{change_pct:+.2f}%")
    cols[2].metric("Risk Level", calculate_risk(change_pct / 100))

    # Backtesting
    st.header("🔍 Backtesting Performance")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    cols = st.columns(2)
    cols[0].metric("RMSE", f"{currency}{rmse:.2f}",
                   "Good" if rmse < 0.02 * latest_price else "Needs improvement")
    cols[1].metric("MAPE", f"{mape:.2f}%",
                   "Excellent" if mape < 2 else "Good" if mape < 5 else "Poor")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual Price'))
    fig.add_trace(go.Scatter(x=y_test.index, y=predictions, name='Predicted Price'))
    fig.update_layout(title="Backtest Results", yaxis_title=f"Price ({currency})")
    st.plotly_chart(fig, use_container_width=True)

    # News Section
    st.header("📰 Latest Stock News")
    news = get_stock_news(ticker)

    if news:
        for item in news[:10]:
            with st.expander(f"{item['date'][:10]} - {item['title']}"):
                if item.get('image'):
                    st.image(item['image'], width=200)
                st.write(f"**{item['publisher']}**")
                if item.get('description'):
                    st.write(item['description'])
                st.write(f"[Read more]({item['url']})")
    else:
        st.warning("No recent news found")


if __name__ == "__main__":
    main()
