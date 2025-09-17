import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="StockSense", layout="wide")
st.title("📈 StockSense – Smart Stock Selection App")

ticker = st.text_input("Enter Stock Ticker (NSE, e.g., INFY.NS):", "INFY.NS").upper()

if ticker:
    try:
        # Validate ticker first
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if "regularMarketPrice" not in info:
            st.error(f"No data found for {ticker}. Please check ticker format (e.g., INFY.NS).")
        else:
            # Fetch 1 year daily data
            data = ticker_obj.history(period="1y")
            if data.empty:
                st.error(f"No price data for {ticker}")
            else:
                close_series = data["Close"].dropna()

                st.subheader(f"Stock Price for {ticker}")
                st.line_chart(close_series)

                # RSI
                rsi_indicator = ta.momentum.RSIIndicator(close_series.values)
                rsi = pd.Series(rsi_indicator.rsi(), index=close_series.index)
                st.subheader("RSI (Relative Strength Index)")
                st.line_chart(rsi)

                # SMA50 & SMA200
                st.subheader("SMA50 & SMA200")
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
        st.error(f"Error fetching data for {ticker}. Please check ticker format (e.g., INFY.NS).")
