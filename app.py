import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="StockSense", layout="wide")
st.title("📈 StockSense – Smart Stock Selection App")

# Input ticker
ticker = st.text_input("Enter Stock Ticker (e.g., INFY.NS):", "INFY.NS").upper()

if ticker:
    try:
        # Fetch data
        data = yf.download(ticker, period="1y", interval="1d")

        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            # Show price chart
            st.subheader(f"Stock Price for {ticker}")
            st.line_chart(data["Close"])

            # Clean Close series
            close_series = data["Close"].dropna()

            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close_series.values)
            rsi = pd.Series(rsi_indicator.rsi(), index=close_series.index)
            st.subheader("RSI (Relative Strength Index)")
            st.line_chart(rsi)

            # Bollinger Bands
            st.subheader("Bollinger Bands")
            bb_indicator = ta.volatility.BollingerBands(close_series)
            bb_upper = pd.Series(bb_indicator.bollinger_hband(), index=close_series.index)
            bb_lower = pd.Series(bb_indicator.bollinger_lband(), index=close_series.index)
            st.line_chart(pd.DataFrame({"Upper Band": bb_upper, "Lower Band": bb_lower, "Close": close_series}))

            # Moving Averages
            st.subheader("Moving Averages (SMA50 & SMA200)")
            sma50 = close_series.rolling(50).mean()
            sma200 = close_series.rolling(200).mean()
            st.line_chart(pd.DataFrame({"Close": close_series, "SMA50": sma50, "SMA200": sma200}))

            # MACD
            st.subheader("MACD")
            macd_indicator = ta.trend.MACD(close_series)
            macd = pd.Series(macd_indicator.macd(), index=close_series.index)
            macd_signal = pd.Series(macd_indicator.macd_signal(), index=close_series.index)
            st.line_chart(pd.DataFrame({"MACD": macd, "Signal": macd_signal}))

    except Exception as e:
        st.error(f"Error fetching data for {ticker}. Please check the ticker format (e.g., INFY.NS).")
