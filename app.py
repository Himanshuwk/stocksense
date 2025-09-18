import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# ------------------------------
# Your Alpha Vantage API Key
# ------------------------------
ALPHA_VANTAGE_API_KEY = "880ISEHBR0L1YIIA"

# ------------------------------
# Fetch Fundamentals from Alpha Vantage
# ------------------------------
def get_fundamentals_alpha_vantage(ticker, api_key=ALPHA_VANTAGE_API_KEY):
    """
    Fetch basic fundamentals (PE, EPS, ROE, Debt-to-Equity) from Alpha Vantage.
    Free tier has limited requests (5 per minute).
    """
    url = f"https://www.alphavantage.co/query"
    
    # Fetch company overview
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return {}
    
    data = response.json()
    if not data or "Symbol" not in data:
        return {}
    
    return {
        "PE": float(data.get("PERatio", 0) or 0),
        "EPS": float(data.get("EPS", 0) or 0),
        "ROE": float(data.get("ReturnOnEquityTTM", 0) or 0) / 100,
        "DebtEquity": float(data.get("DebtEquity", 0) or 0)
    }

# ------------------------------
# Fetch Technical Indicators from yfinance
# ------------------------------
def get_technicals(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    if df.empty:
        return {}

    close = df["Close"]

    # RSI calculation
    delta = close.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = round(rsi.iloc[-1], 2)

    # Moving averages
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    return {
        "RSI": latest_rsi,
        "MA50": round(ma50, 2),
        "MA200": round(ma200, 2)
    }

# ------------------------------
# Scoring logic
# ------------------------------
def calculate_stock_score(fundamentals, technicals):
    fund_score = 0
    tech_score = 0

    # Fundamental scoring
    if fundamentals.get("ROE", 0) > 0.12:
        fund_score += 20
    if fundamentals.get("PE", 0) < 25 and fundamentals.get("PE", 0) > 0:
        fund_score += 20
    if fundamentals.get("EPS", 0) > 5:
        fund_score += 20
    if fundamentals.get("DebtEquity", 1) < 1:
        fund_score += 20

    # Technical scoring
    if 30 < technicals.get("RSI", 50) < 70:
        tech_score += 20
    if technicals.get("MA50", 0) > technicals.get("MA200", 0):
        tech_score += 20

    final_score = (fund_score + tech_score) / 1.2  # Normalize to 100
    return round(final_score, 2), fund_score, tech_score

# ------------------------------
# Streamlit App
# ------------------------------
st.title("📊 StockSense – Smart Stock Selection App")
st.write("A simple, user-friendly tool to combine **fundamental** and **technical** analysis.")

ticker = st.text_input("Enter Stock Ticker (e.g., INFY.BSE or RELIANCE.BSE for India, AAPL for US):")

if ticker:
    try:
        with st.spinner("Fetching data..."):
            fundamentals = get_fundamentals_alpha_vantage(ticker)
            technicals = get_technicals(ticker)

        if not fundamentals:
            st.warning("⚠️ Could not fetch fundamentals. Check API key and ticker.")
        elif not technicals:
            st.warning("⚠️ Could not fetch technicals. Check ticker format.")
        else:
            final_score, fund_score, tech_score = calculate_stock_score(fundamentals, technicals)

            st.subheader(f"Results for {ticker}")
            st.metric("📊 Final Stock Score", f"{final_score} / 100")

            st.write("### 🔎 Fundamental Data")
            st.json(fundamentals)

            st.write("### 📈 Technical Data")
            st.json(technicals)

    except Exception as e:
        st.error(f"Error: {e}")
