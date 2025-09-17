import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="StockSense", layout="wide")

st.title("📈 StockSense – Smart Stock Selection App")

# Stock input
ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS for Infosys):", "INFY.NS")

if ticker:
    # Download stock data
    data = yf.download(ticker, period="1y", interval="1d")

    if not data.empty:
        st.subheader(f"Stock Price for {ticker}")
        st.line_chart(data["Close"])

        # RSI indicator
        rsi = ta.momentum.RSIIndicator(data["Close"]).rsi()
        st.subheader("RSI (Relative Strength Index)")
        st.line_chart(rsi)
    else:
        st.error("No data found. Please check the ticker symbol.")
