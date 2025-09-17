import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import openai
from dotenv import load_dotenv
import os

# --- Load Environment Variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Functions ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if "regularMarketPrice" not in info or not info["regularMarketPrice"]:
            return None, None, f"No data found for {ticker}."

        data = ticker_obj.history(period="1y")
        if data.empty:
            return None, None, f"No price data for {ticker}."

        close_series = data["Close"].dropna()

        # Technical indicators
        data["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()
        data["MACD"] = ta.trend.MACD(close_series).macd_diff()
        data["SMA50"] = close_series.rolling(window=50).mean()
        data["SMA200"] = close_series.rolling(window=200).mean()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close_series)
        data["BB_High"] = bb.bollinger_hband()
        data["BB_Low"] = bb.bollinger_lband()

        fundamentals = get_fundamental_data(ticker)

        return data, fundamentals, None
    except Exception as e:
        return None, None, f"Error fetching data for {ticker}: {e}"

def get_fundamental_data(ticker_str):
    try:
        info = yf.Ticker(ticker_str).info
        metrics = {
            'Promoter Holding': info.get('sharesOutstanding', None),
            'Debt/Equity': info.get('debtToEquity', None),
            'ROE': info.get('returnOnEquity', None),
            'P/E Ratio': info.get('forwardPE', None) or info.get('trailingPE', None),
        }
        return metrics
    except:
        return {'Promoter Holding': None, 'Debt/Equity': None, 'ROE': None, 'P/E Ratio': None}

def calculate_stock_score(fundamentals, technicals):
    fund_score, tech_score = 0, 0

    debt_equity = fundamentals.get('Debt/Equity')
    roe = fundamentals.get('ROE')
    rsi = technicals['RSI'][-1]
    close = technicals['Close'][-1]
    sma50 = technicals['SMA50'][-1]

    if debt_equity is not None and debt_equity < 1:
        fund_score += 20
    if roe is not None and roe > 0.12:
        fund_score += 20

    if rsi is not None and 30 < rsi < 70:
        tech_score += 20
    if close is not None and sma50 is not None and close > sma50:
        tech_score += 20

    final_score = (fund_score * 0.6) + (tech_score * 0.4)
    return final_score, fund_score, tech_score

def get_score_label(final_score):
    if final_score >= 80:
        return "🟢 Strong Buy"
    elif final_score >= 50:
        return "🟡 Watchlist"
    else:
        return "🔴 Avoid"

def get_ai_insights(ticker, fundamentals, technicals, final_score):
    prompt = f"""
    Analyze the stock {ticker} for a beginner investor:
    - Fundamentals: {fundamentals}
    - Technicals: RSI {technicals['RSI'][-1]:.2f}, Close {technicals['Close'][-1]:.2f}, SMA50 {technicals['SMA50'][-1]:.2f}
    - Score: {final_score:.2f}/100
    Give a simple 2-3 sentence summary. Recommend Strong Buy / Watchlist / Avoid.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI insights: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="StockSense", layout="wide")
st.title("📈 StockSense – Smart Stock Selection App")
st.markdown("Analyze any NSE stock dynamically with fundamental & technical indicators.")

ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS):", "INFY.NS").upper()

if st.button("Analyze Stock") and ticker:
    with st.spinner("Analyzing stock..."):
        data, fundamentals, error_message = get_stock_data(ticker)

    if error_message:
        st.error(error_message)
    else:
        if not all(fundamentals.values()):
            st.warning("Fundamental data is incomplete. Score may not be accurate.")

        technicals = {
            'Close': data["Close"],
            'RSI': data["RSI"],
            'MACD': data["MACD"],
            'SMA50': data["SMA50"],
        }

        final_score, fund_score, tech_score = calculate_stock_score(fundamentals, technicals)
        score_label = get_score_label(final_score)
        ai_insights = get_ai_insights(ticker, fundamentals, technicals, final_score)

        st.subheader(f"Analysis for {ticker}")
        st.metric(label="StockSense Score", value=f"{final_score:.2f} / 100", delta=score_label)
        st.info(ai_insights)

        tab1, tab2 = st.tabs(["Charts & Technicals", "Fundamental Metrics"])
        with tab1:
            st.subheader("Technical Analysis")
            st.line_chart(pd.DataFrame({
                "Close": data["Close"],
                "SMA50": data["SMA50"],
                "SMA200": data["SMA200"],
                "BB_High": data["BB_High"],
                "BB_Low": data["BB_Low"]
            }))
            st.line_chart(data["RSI"])
            st.line_chart(data["MACD"])
            st.line_chart(data["Volume"])
        with tab2:
            st.subheader("Key Fundamental Ratios")
            fund_df = pd.DataFrame(fundamentals, index=["Value"]).T
            st.dataframe(fund_df)
            st.write("*(Note: Some fundamental metrics may be limited. Placeholder data may appear.)*")
