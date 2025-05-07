import yfinance as yf
import pandas as pd
import feedparser
from textblob import TextBlob
import time

# ========== Step 1: Define NSE stocks (NIFTY 50 example) ==========
nifty_50 = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'BEL.NS', 'TATAMOTORS.NS', 
    'LT.NS', 'TITAN.NS', 'NATIONALUM.NS', 'TATAPOWER.NS', 'RAILTEL.NS'
]

# ========== Step 2: Function to fetch fundamental data ==========
def get_fundamentals(stock):
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info

        return {
            'Symbol': stock,
            'PE Ratio': info.get('trailingPE'),
            'EPS': info.get('trailingEps'),
            'ROE': info.get('returnOnEquity'),
            'Market Cap': info.get('marketCap'),
            'Debt/Equity': info.get('debtToEquity'),
            'Revenue Growth': info.get('revenueGrowth')
        }
    except Exception as e:
        print(f"[ERROR] {stock}: {e}")
        return None

# ========== Step 3: Function to fetch news sentiment ==========
def get_news_sentiment(stock_name):
    query = stock_name.replace('.NS', '') + " stock"
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"

    feed = feedparser.parse(url)
    sentiments = []

    for entry in feed.entries[:5]:  # limit to 5 latest headlines
        title = entry.title
        blob = TextBlob(title)
        sentiments.append(blob.sentiment.polarity)

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0  # neutral if no news

    time.sleep(1)  # avoid spamming Google
    return avg_sentiment

# ========== Step 4: Screener Logic ==========
results = []

for stock in nifty_50:
    print(f"Processing {stock}...")
    fundamentals = get_fundamentals(stock)
    if fundamentals:
        sentiment = get_news_sentiment(stock)
        fundamentals['News Sentiment'] = sentiment
        results.append(fundamentals)

# ========== Step 5: Create DataFrame & Apply Filters ==========
df = pd.DataFrame(results)

# Apply screening conditions
filtered_df = df[
    (df['ROE'] > 0.15) &
    (df['PE Ratio'] < 30) &
    (df['Debt/Equity'] < 1.5) &
    (df['Revenue Growth'] > 0) &
    (df['News Sentiment'] > -0.2)
]

# Save result
filtered_df.to_csv("selected_stocks_with_sentiment.csv", index=False)
print("\n✅ Screener completed. Top stocks:")
print(filtered_df)
