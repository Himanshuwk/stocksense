import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import openai
from dotenv import load_dotenv
import os

# --- 0. Load Environment Variables (for API Key) ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 1. Functions for Data & Analysis ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Fetches stock data and returns a DataFrame, fundamentals, and a status."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Check if the ticker exists and has price data
        if "regularMarketPrice" not in info or not info["regularMarketPrice"]:
            return None, None, f"No data found for {ticker}. Please check ticker format."
        
        # Fetch 1 year daily data
        data = ticker_obj.history(period="1y")
        if data.empty:
            return None, None, f"No price data for {ticker}"
            
        close_series = data["Close"].dropna()

        # Add technical indicators
        data["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()
        data["MACD"] = ta.trend.MACD(close_series).macd_diff()
        data["SMA50"] = close_series.rolling(window=50).mean()
        data["SMA200"] = close_series.rolling(window=200).mean()
        
        # Fetch fundamental data using ticker string
        fundamentals = get_fundamental_data(ticker)
        
        return data, fundamentals, None
    except Exception as e:
        return None, None, f"Error fetching data for {ticker}: {e}"

def get_fundamental_data(ticker_str):
    """Fetches key fundamental metrics using ticker string."""
    try:
        ticker_obj = yf.Ticker(ticker_str)
        info = ticker_obj.info
        metrics = {
            'Promoter Holding': info.get('sharesOutstanding', None),  # Placeholder
            'Debt/Equity': info.get('debtToEquity', None),
            'ROE': info.get('returnOnEquity', None),
            'P/E Ratio': info.get('forwardPE', None) or info.get('trailingPE', None),
        }
        return metrics
    except:
        # Return empty metrics if any error occurs
        return {'Promoter Holding': None, 'Debt/Equity': None, 'ROE': None, 'P/E Ratio': None}

def calculate_stock_score(fundamentals, technicals):
    """Calculates a composite score based on weighted criteria safely."""
    fund_score = 0
    tech_score = 0

    debt_equity = fundamentals.get('Debt/Equity')
    roe = fundamentals.get('ROE')

    if debt_equity is not None and debt_equity < 1:
        fund_score += 20
    if roe is not None and roe > 0.12:
        fund_score += 20

    rsi = technicals['RSI'][-1]
    close = technicals['Close'][-1]
    sma50 = technicals['SMA50'][-1]

    if rsi is not None and 30 < rsi < 70:
        tech_score += 20
    if close is not None and sma50 is not None and close > sma50:
        tech_score += 20

    final_score = (fund_score * 0.6) + (tech_score * 0.4)
    return final_score, fund_score, tech_score

def get_ai_insights(ticker, fundamentals, technicals, final_score):
    """Generates plain-language insights using OpenAI LLM."""
    prompt = f"""
    Analyze the stock {ticker} for a beginner investor based on the following data:
    - Fundamental Metrics: {fundamentals}
    - Technical Indicators: Current RSI is {technicals['RSI'][-1]:.2f},
      Last Close Price is {technicals['Close'][-1]:.2f},
      50-day SMA is {technicals['SMA50'][-1]:.2f}.
    - Overall Score: {final_score:.2f}/100

    Provide a simple, plain-language summary in 2-3 sentences. Explain what the data means without using jargon.
    For example, for RSI, explain what "overbought" or "oversold" means in simple terms. 
    Tell the investor if the stock is a 'Strong Buy', 'Watchlist', or 'Avoid' based on the score (e.g., >80: Strong Buy, 50-80: Watchlist, <50: Avoid).
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI insights: {e}"

# --- 2. Streamlit UI ---
st.set_page_config(page_title="StockSense", layout="wide")
st.title("📈 StockSense – Smart Stock Selection App")
st.markdown("Analyze any NSE stock dynamically with fundamental & technical indicators.")

ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS):", "INFY.NS").upper()

if st.button("Analyze Stock"):
    if ticker:
        with st.spinner("Analyzing stock..."):
            data, fundamentals, error_message = get_stock_data(ticker)
        
        if error_message:
            st.error(error_message)
        else:
            if not all(fundamentals.values()):
                st.warning("Warning: Fundamental data is incomplete. Score may not be accurate.")
            
            technicals = {
                'Close': data["Close"],
                'RSI': data["RSI"],
                'MACD': data["MACD"],
                'SMA50': data["SMA50"],
            }

            final_score, fund_score, tech_score = calculate_stock_score(fundamentals, technicals)
            ai_insights = get_ai_insights(ticker, fundamentals, technicals, final_score)

            st.subheader(f"Analysis for {ticker}")
            st.metric(label="StockSense Score", value=f"{final_score:.2f} / 100")
            st.info(ai_insights)

            # Tabs for charts and fundamentals
            tab1, tab2 = st.tabs(["Charts & Technicals", "Fundamental Metrics"])
            
            with tab1:
                st.subheader("Technical Analysis")
                st.line_chart(pd.DataFrame({
                    "Close": data["Close"], 
                    "SMA50": data["SMA50"], 
                    "SMA200": data["SMA200"]
                }))
                st.line_chart(data["RSI"])
                st.line_chart(data["MACD"])
                
            with tab2:
                st.subheader("Key Fundamental Ratios")
                fund_df = pd.DataFrame(fundamentals, index=["Value"]).T
                st.dataframe(fund_df)
                st.write("*(Note: Some fundamental metrics may be limited. This is a placeholder.)*")
    else:
        st.error("Please enter a stock ticker.")
