import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="StockSense", layout="wide")

st.title("📈 StockSense – Smart Stock Selection App")

# Input stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS):", "INFY.NS")

if ticker:
    data = yf.download(ticker, period="1y", interval="1d")
    st.subheader(f"Data for {ticker}")
    st.line_chart(data["Close"])

    # Example technical indicator
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    st.line_chart(data["RSI"])

