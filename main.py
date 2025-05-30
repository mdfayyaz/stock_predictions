import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import requests
import urllib.parse
from transformers import pipeline


# Streamlit App Title
st.markdown(f'<h1 style="font-size: 30px; color: #1C39BB;">📈 Stock OHLC Prediction App</h1>', unsafe_allow_html=True)
# Select stock and date range
stock_symbol = st.text_input("Enter NSE stock symbol (e.g. BEL.NS):", "BEL.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if start_date >= end_date:
    st.warning("⚠️ End date must be after start date.")
    st.stop()

# Download historical data
raw_df = yf.download(stock_symbol, start=start_date, end=end_date + pd.Timedelta(days=1))

if raw_df.empty:
    st.warning("⚠️ No data found. Please check the stock symbol.")
    st.stop()

st.markdown(f'<h1 style="font-size: 20px; color: #1C39BB;">Showing data for {stock_symbol}</h1>', unsafe_allow_html=True)
st.dataframe(raw_df.tail())

# Feature Engineering
df = raw_df.copy()
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=5).std()
df['RSI'] = 100 - (100 / (1 + df['Returns'].rolling(window=14).mean() / df['Returns'].rolling(window=14).std()))
df['Bollinger Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['Bollinger Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
df.dropna(inplace=True)

# Trend Labels Based on Indicators
df['Trend_MA'] = np.where(df['MA5'] > df['MA10'], 1, 0)
df['Trend_RSI'] = np.where(df['RSI'] > 50, 1, 0)
df = df[df['Bollinger Upper'].notnull() & df['Bollinger Lower'].notnull()]

boll_upper = df['Bollinger Upper'].values.flatten()
boll_lower = df['Bollinger Lower'].values.flatten()
close_vals = df['Close'].values.flatten()

trend_bollinger = np.where(close_vals > boll_upper, 1,
                   np.where(close_vals < boll_lower, 0, 1))

df['Trend_Bollinger'] = trend_bollinger

trend_features = ['Trend_MA', 'Trend_RSI', 'Trend_Bollinger']
df['Overall_Trend'] = df[trend_features].mean(axis=1).round().astype(int)

# Features for ML
feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'Returns', 'Volatility', 'RSI']
X = df[feature_columns]

# Targets: Open, High, Low, Close
ohlc_targets = ['Open', 'High', 'Low', 'Close']
y_ohlc = df[ohlc_targets].loc[X.index]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MultiOutput Regressor
model = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42))
split = int(len(X_scaled) * 0.8)
model.fit(X_scaled[:split], y_ohlc.iloc[:split])
y_ohlc_pred = model.predict(X_scaled[split:])

# Predict next day's OHLC
X_last_scaled = scaler.transform(df[feature_columns].iloc[[-1]])
predicted_ohlc = model.predict(X_last_scaled)[0]

# Yesterday's OHLC
yesterday_ohlc = raw_df[ohlc_targets].iloc[-2].values

# Latest OHLC (today's actual)
today_ohlc = raw_df[ohlc_targets].iloc[-1].values

# Display comparison table
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🔍 Yesterday vs Today vs Predicted (Next Day) OHLC for {stock_symbol}</h2>', unsafe_allow_html=True)

comparison_df = pd.DataFrame({
    'Yesterday': yesterday_ohlc.flatten(),
    'Today': today_ohlc.flatten(),
    'Predicted (Next Day)': predicted_ohlc.flatten()
}, index=ohlc_targets)

st.dataframe(comparison_df)

# Indicator Summary Table
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🧭 Indicator Trends Table for {stock_symbol}</h2>', unsafe_allow_html=True)
indicator_table = pd.DataFrame({
    'MA Trend': ['Uptrend' if df['Trend_MA'].iloc[-1] else 'Downtrend'],
    'RSI Trend': ['Bullish' if df['Trend_RSI'].iloc[-1] else 'Bearish'],
    'Bollinger Trend': ['Above Upper Band' if df['Trend_Bollinger'].iloc[-1] == 1 else ('Below Lower Band' if df['Trend_Bollinger'].iloc[-1] == 0 else 'Within Band')],
    'ML Trend Prediction': ['Uptrend' if df['Overall_Trend'].iloc[-1] else 'Downtrend']
})
st.dataframe(indicator_table)

# Show current trend based on MA crossover
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📊 Current ML-Based Market Trend for {stock_symbol}"</h2>', unsafe_allow_html=True)

st.write(f"The market for {stock_symbol} is predicted to be in an **{'Uptrend' if df['Overall_Trend'].iloc[-1] else 'Downtrend'}**.")

# Actual vs Predicted Close Plot
actual_close = y_ohlc['Close'].iloc[split:].values.flatten()
pred_close = y_ohlc_pred[:, 3].flatten()

plot_df = pd.DataFrame({
    'Actual': actual_close,
    'Predicted': pred_close
}, index=y_ohlc.iloc[split:].index)

st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📉 Actual vs Predicted Close Price for {stock_symbol}</h2>', unsafe_allow_html=True)

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
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">💰 Actual vs Predicted Close Price for Last 7 Working Days for {stock_symbol}</h2>', unsafe_allow_html=True)
st.dataframe(last_7_days_data)
# Backtesting Metrics
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📈 Backtesting Summary for {stock_symbol}</h2>', unsafe_allow_html=True)

mae = mean_absolute_error(actual_close, pred_close)
mse = mean_squared_error(actual_close, pred_close)
rmse = np.sqrt(mse)
r2 = r2_score(actual_close, pred_close)

backtest_metrics = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Value': [mae, mse, rmse, r2]
})

st.dataframe(backtest_metrics.style.format({"Value": "{:.4f}"}))

# User-Friendly Summary of Backtesting with Accuracy Estimate
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📘 What Do These Metrics Mean for {stock_symbol}?</h2>', unsafe_allow_html=True)
st.markdown(f"""
- **MAE (Mean Absolute Error)**: On average, the model's predictions for {stock_symbol} were ₹{mae:.2f} away from the actual closing prices.
- **RMSE (Root Mean Squared Error)**: This penalizes larger errors more than MAE. A lower value means better accuracy.
- **R² Score**: This tells how well the model explains the price movements for {stock_symbol}. A value closer to 1 means a better fit. 
  - **Accuracy Estimate**: Based on the R² score of {r2:.4f}, the model is approximately **{r2 * 100:.2f}% accurate** in predicting the closing prices for {stock_symbol}.
""")

# Additional insights and user-friendly summary
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🧠 Additional Insights & Sentiment Analysis for {stock_symbol}</h2>', unsafe_allow_html=True)

# News Sentiment Analysis
# 📰 News & FinBERT Sentiment Analysis Section
st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📰 News & Sentiment Analysis for {stock_symbol}</h2>', unsafe_allow_html=True)


API_KEY = st.secrets["newsapi"]["api_key"]  # Replace with your actual NewsAPI key

# FinBERT setup
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

finbert = load_finbert()

# Get stock news function (using company name now!)
def get_stock_news(stock_symbol, api_key):
    company_mapping = {
        'BEL.NS': 'Bharat Electronics Ltd',
        'RAILTEL.NS': 'RailTel Corporation of India Ltd.Indian Navaratna Public Sector',
        'TATAMOTORS.NS': 'Tata Motors',
        'TATAPOWER.NS': 'Tata Power',
        'TCS.NS': 'Tata Consultancy Services',
        'INFY.NS': 'Infosys',
        'RELIANCE.NS': 'Reliance Industries',
        'BDL.NS': 'Bharat Dynamics Ltd',
        'NATIONALUM.NS': 'National Aluminium Company',
        'DLF.NS': 'DLF Limited',
        'TITAN.NS': 'Titan Company Ltd',
        'BPCL.NS': 'Bharat Petroleum Corporation Limited'

        # add more mappings as needed
    }
    company_name = company_mapping.get(stock_symbol.upper(), stock_symbol)

    query = urllib.parse.quote(company_name)

    # 🎯 Fetch recent news only (last 7 days)
    today = pd.to_datetime("today").date()
    seven_days_ago = today - pd.Timedelta(days=7)

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"from={seven_days_ago}&to={today}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"pageSize=10&"
        f"apiKey={api_key}"
    )

    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return [article['title'] for article in news_data['articles']]
    else:
        return []

# FinBERT sentiment analyzer
def analyze_sentiment(news_headlines):
    results = []
    for headline in news_headlines:
        result = finbert(headline)[0]
        results.append({
            'headline': headline,
            'sentiment': result['label'].capitalize(),  # Positive / Negative / Neutral
            'score': result['score']
        })
    return results

try:
    stock_news = get_stock_news(stock_symbol, API_KEY)

    if stock_news:
        st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">📰 Latest News Headlines for {stock_symbol}</h2>', unsafe_allow_html=True)
        for news in stock_news:
            st.write(f"- {news}")

        # Analyze with FinBERT
        sentiment_results = analyze_sentiment(stock_news)

        # Display sentiment for each headline
        st.markdown(f'<h2 style="font-size: 24px; color: #1C39BB;">🧠 Sentiment Analysis of News Headlines</h2>',
                    unsafe_allow_html=True)


        for item in sentiment_results:
            st.write(f"**{item['sentiment']}** ({item['score']:.2f}): {item['headline']}")

        # Aggregate counts
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for item in sentiment_results:
            sentiment_counts[item['sentiment']] += 1

        # Display overall
        total = sum(sentiment_counts.values())
        if total > 0:
            overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            st.write(f'<h2 style="font-size: 24px; color: #1C39BB;"> 📊 Overall Sentiment for {stock_symbol}: **{overall_sentiment}**</h2>', unsafe_allow_html=True)


        else:
            st.write(
                f'<h2 style="font-size: 24px; color: #1C39BB;"> "No news available to analyze for {stock_symbol}.</h2>',
                unsafe_allow_html=True)


        # Beautiful bar chart 📊

        st.write(
            f'<h2 style="font-size: 24px; color: #1C39BB;"> 📈 Sentiment Distribution Chart for {stock_symbol}.</h2>',
            unsafe_allow_html=True)



        import plotly.express as px

        fig = px.bar(
            x=list(sentiment_counts.keys()),
            y=list(sentiment_counts.values()),
            labels={'x': 'Sentiment', 'y': 'Number of Articles'},
            title=f"Sentiment Breakdown for {stock_symbol}",
            color=list(sentiment_counts.keys()),
            color_discrete_map={
                "Positive": "green",
                "Negative": "red",
                "Neutral": "blue"
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"No news found for {stock_symbol}.")

except Exception as e:
    st.error(f"⚠️ Error fetching or analyzing news for {stock_symbol}: {e}")


# Plain Summary for Non-Technical Users
st.write(
            f'<h2 style="font-size: 24px; color: #1C39BB;">📊 Easy-to-Understand Summary for {stock_symbol}</h2>',
            unsafe_allow_html=True)

st.info(f"Coming soon: Plain-language insights for {stock_symbol}, including potential risk levels and trend summaries for non-technical users.")
