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
    """Fetches stock data and returns a ticker object."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if "regularMarketPrice" not in info:
            return None, "No data found for this ticker."
        
        # Fetch 1 year daily data
        data = ticker_obj.history(period="1y")
        if data.empty:
            return None, "No price data for this ticker."
            
        # Add technical indicators
        data["RSI"] = ta.momentum.rsi(data["Close"])
        data["MACD"] = ta.trend.macd_diff(data["Close"])
        data["SMA50"] = data["Close"].rolling(window=50).mean()
        data["SMA200"] = data["Close"].rolling(window=200).mean()
        
        return ticker_obj, data
    except Exception as e:
        return None, f"Error fetching data: {e}"

def get_fundamental_data(ticker_obj):
    """Fetches key fundamental metrics."""
    info = ticker_obj.info
    metrics = {
        'Promoter Holding': info.get('sharesOutstanding', 0) / info.get('sharesOutstanding', 1) if info.get('sharesOutstanding') else None,
        'Debt/Equity': info.get('debtToEquity', None),
        'ROE': info.get('returnOnEquity', None),
        'P/E Ratio': info.get('forwardPE', None) or info.get('trailingPE', None),
    }
    # Note: Promoter holding data is not reliable on yfinance. You'll need a different source for production.
    # The current code is a placeholder to demonstrate the logic.
    return metrics

def calculate_stock_score(fundamentals, technicals):
    """Calculates a composite score based on the weighted criteria."""
    fund_score = 0
    tech_score = 0

    # Fundamental Scoring
    # Example logic: add points for good financials
    if fundamentals.get('Debt/Equity', 2) < 1: fund_score += 20
    if fundamentals.get('ROE', 0) > 0.12: fund_score += 20
    # Add more rules as needed...

    # Technical Scoring
    # Example logic: add points for positive signals
    if technicals['RSI'][-1] > 30 and technicals['RSI'][-1] < 70: tech_score += 20
    if technicals['Close'][-1] > technicals['SMA50'][-1]: tech_score += 20
    
    # Combine scores with your weighting (60/40)
    final_score = (fund_score * 0.6) + (tech_score * 0.4)
    
    return final_score, fund_score, tech_score

def get_ai_insights(ticker, fundamentals, technicals, final_score):
    """Generates plain-language insights using an LLM."""
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

# --- 2. Streamlit UI & Logic ---
st.set_page_config(page_title="StockSense", layout="wide")
st.title("📈 StockSense – Smart Stock Selection App")
st.markdown("A simple, user-friendly tool to combine fundamental and technical analysis.")

ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS):", "INFY.NS").upper()

if st.button("Analyze Stock"):
    if ticker:
        ticker_obj, data_status = get_stock_data(ticker)
        
        if ticker_obj is None:
            st.error(data_status)
        else:
            with st.spinner("Analyzing stock..."):
                # Two-Layer Screening
                fundamentals = get_fundamental_data(ticker_obj)
                
                # Check for critical fundamental data
                if not all(fundamentals.values()):
                    st.warning("Warning: Fundamental data is incomplete. Score may not be accurate.")
                
                # Get the latest technical data
                technicals = {
                    'Close': data["Close"],
                    'RSI': data["RSI"],
                    'MACD': data["MACD"],
                    'SMA50': data["SMA50"],
                }
                
                # Scoring
                final_score, fund_score, tech_score = calculate_stock_score(fundamentals, technicals)
                
                # AI Insights
                ai_insights = get_ai_insights(ticker, fundamentals, technicals, final_score)
            
            # Display Results
            st.subheader(f"Analysis for {ticker}")
            st.metric(label="StockSense Score", value=f"{final_score:.2f} / 100")
            
            st.info(ai_insights)
            
            # Tabs for more detail
            tab1, tab2 = st.tabs(["Charts & Technicals", "Fundamental Metrics"])
            
            with tab1:
                st.subheader("Technical Analysis")
                st.line_chart(pd.DataFrame({"Close": data["Close"], "SMA50": data["SMA50"], "SMA200": data["SMA200"]}))
                st.line_chart(data["RSI"])
                
            with tab2:
                st.subheader("Key Fundamental Ratios")
                fund_df = pd.DataFrame(fundamentals, index=["Value"]).T
                st.dataframe(fund_df)
                st.write("*(Note: Promoter Holding data from yfinance may be limited. This is a placeholder.)*")
    else:
        st.error("Please enter a stock ticker.")
