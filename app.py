import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import openai
from dotenv import load_dotenv
import plotly.express as px

# ------------------------------
# Config / Load API key
# ------------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

st.set_page_config(page_title="StockSense (MVP)", layout="wide")
st.title("📈 StockSense — Smart Stock Selection")
st.markdown("Personalized stock scoring + AI insights for Indian stocks. Use NSE tickers like `RELIANCE.NS`, `INFY.NS`, `TCS.NS`.")

# Initialize session state for button clicks if not already present
if 'analysis_mode' not in st.session_state:
    st.session_state['analysis_mode'] = 'individual'

# ------------------------------
# Helper functions
# ------------------------------
@st.cache_data(ttl=3600)
def fetch_price_data(ticker, period="1y", interval="1d"):
    """Fetch price history and return DataFrame or None on failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_info(ticker):
    """Fetch yfinance info (may be partial)."""
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}

def compute_technical_indicators(df):
    """Add RSI, SMA50, SMA200, MACD diff, Bollinger bands, returns, volatility."""
    
    close = df["Close"].dropna().squeeze()
    
    if len(close) < 200: 
        st.warning("Not enough historical data to compute all indicators.")
        result = pd.DataFrame(index=df.index)
        result["RSI"] = np.nan
        result["SMA50"] = np.nan
        result["SMA200"] = np.nan
        result["MACD"] = np.nan
        result["BB_high"] = np.nan
        result["BB_low"] = np.nan
        result["Volume"] = np.nan if "Volume" in df.columns else np.nan
        result["Return"] = np.nan
        return result

    result = pd.DataFrame(index=close.index)
    result["Close"] = close
    
    result["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    result["SMA50"] = close.rolling(50).mean()
    result["SMA200"] = close.rolling(200).mean()
    result["MACD"] = ta.trend.MACD(close).macd_diff()
    bb = ta.volatility.BollingerBands(close)
    result["BB_high"] = bb.bollinger_hband()
    result["BB_low"] = bb.bollinger_lband()
    
    if "Volume" in df.columns:
        result["Volume"] = df["Volume"]
    
    result["Return"] = close.pct_change()
    
    return result

def safe_get(dct, key, default=None):
    v = dct.get(key, default)
    if pd.isna(v) or v in (None, "", "None"):
      return default
    return v

def calc_beta(series_stock, series_market):
    """Calculate beta of stock vs market using aligned daily returns."""
    s = pd.concat([series_stock, series_market], axis=1).dropna()
    if s.shape[0] < 30:
        return None
    cov = s.iloc[:,0].cov(s.iloc[:,1])
    var = s.iloc[:,1].var()
    if var == 0 or math.isnan(cov) or math.isnan(var):
        return None
    return cov / var

def score_stock(fundamentals, tech_latest, weight_fund=0.6):
    """Score stock: combine fundamentals and technicals with user weights. Returns (final_score, details)."""
    fund_points = 0
    de = fundamentals.get("de_ratio")
    if de is not None:
        try:
            if de < 1:
                fund_points += 20
        except:
            pass
    roe = fundamentals.get("roe")
    if roe is not None:
        try:
            if roe > 0.12:
                fund_points += 20
        except:
            pass
    pe = fundamentals.get("pe")
    if pe is not None and pe > 0:
        if pe < 30:
            fund_points += 20
    prom = fundamentals.get("promoter_holding")
    if prom is not None:
        try:
            if prom > 0.5:
                fund_points += 20
        except:
            pass

    tech_points = 0
    rsi = tech_latest.get("RSI")
    price = tech_latest.get("Close")
    sma50 = tech_latest.get("SMA50")
    if rsi is not None and not pd.isna(rsi):
        if 30 < rsi < 70:
            tech_points += 20
    if price is not None and sma50 is not None and not pd.isna(sma50):
        if price > sma50:
            tech_points += 20

    fund_score_norm = min(100, (fund_points / 80) * 100) if fund_points >= 0 else 0
    tech_score_norm = min(100, (tech_points / 40) * 100) if tech_points >= 0 else 0

    final_score = weight_fund * fund_score_norm + (1 - weight_fund) * tech_score_norm

    details = {
        "raw_fund_points": fund_points,
        "raw_tech_points": tech_points,
        "fund_score_norm": round(fund_score_norm,2),
        "tech_score_norm": round(tech_score_norm,2),
    }
    return round(final_score,2), details

def ai_summary_safe(ticker, fundamentals, tech_latest, final_score):
    """Return an AI-generated 2-line summary if API key present, else a simple rule-based summary."""
    prompt = (
        f"Explain to a beginner investor in 2-3 simple sentences the situation for {ticker}.\n"
        f"Fundamentals: {fundamentals}\n"
        f"Technicals: RSI={tech_latest.get('RSI')}, Close={tech_latest.get('Close')}, SMA50={tech_latest.get('SMA50')}\n"
        f"Overall score: {final_score} out of 100.\n"
        "Keep it short, no jargon. At the end, give one sentence recommendation: Strong Buy / Watchlist / Avoid.\n"
        "Include a 1-line explanation why."
    )
    if not OPENAI_KEY:
        rsi = tech_latest.get("RSI")
        rec = "Watchlist"
        if final_score >= 80:
            rec = "Strong Buy"
        elif final_score < 50:
            rec = "Avoid"
        s = f"{ticker}: Score {final_score}/100. Recommendation: {rec}. "
        if rsi is not None:
            s += f"RSI {rsi:.1f} shows {'neutral' if 30<rsi<70 else ('overbought' if rsi>=70 else 'oversold')}."
        return s

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        txt = response.choices[0].message.content.strip()
        return txt
    except Exception as e:
        return f"AI unavailable: {e}"

def analyze_individual_stock(ticker, weight_fund, show_tooltips):
    """Main function to run the individual stock analysis."""
    df = fetch_price_data(ticker, period=st.session_state.lookback_period)
    info = fetch_info(ticker)
    
    if df is None or info is None:
        st.error("No data found for ticker. Check format (e.g., INFY.NS).")
    else:
        tech = compute_technical_indicators(df)
        
        if tech.empty or pd.isna(tech.get("RSI", pd.Series([np.nan]))).iloc[-1]:
            st.warning("Not enough data to calculate all indicators. Please try a longer lookback period or a different ticker.")
        else:
            tech_latest = {
                "Close": float(tech["Close"].iloc[-1]) if not tech["Close"].isna().all() else None,
                "RSI": float(tech["RSI"].iloc[-1]) if "RSI" in tech.columns else None,
                "SMA50": float(tech["SMA50"].iloc[-1]) if "SMA50" in tech.columns else None,
                "SMA200": float(tech["SMA200"].iloc[-1]) if "SMA200" in tech.columns else None,
            }

            fundamentals = {
                "pe": safe_get(info, "trailingPE", None) or safe_get(info, "forwardPE", None),
                "roe": safe_get(info, "returnOnEquity", None),
                "de_ratio": safe_get(info, "debtToEquity", None),
                "promoter_holding": None
            }
            sector = safe_get(info, "sector", "Unknown")

            final_score, details = score_stock(fundamentals, tech_latest, weight_fund)
            label = "🟢 Strong Buy" if final_score>=80 else ("🟡 Watchlist" if final_score>=50 else "🔴 Avoid")

            st.subheader(f"{ticker} — {sector}")
            st.metric("StockSense score", f"{final_score}/100", delta=label)

            if show_tooltips:
                st.caption("Score = weighted blend of fundamentals & technicals. Adjust weight in the sidebar.")

            fig_df = pd.DataFrame({
                "Close": tech["Close"], "SMA50": tech["SMA50"], "SMA200": tech["SMA200"],
                "BB_high": tech.get("BB_high"), "BB_low": tech.get("BB_low")
            })
            fig = px.line(fig_df.reset_index(), x=fig_df.index, y=fig_df.columns,
                          labels={"value":"Price", "index":"Date"}, title=f"{ticker} Price & Indicators")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("RSI & MACD")
            rsi_fig = px.line(tech.reset_index(), x=tech.index, y="RSI", title="RSI")
            macd_fig = px.line(tech.reset_index(), x=tech.index, y="MACD", title="MACD")
            st.plotly_chart(rsi_fig, use_container_width=True)
            st.plotly_chart(macd_fig, use_container_width=True)

            st.subheader("Key fundamentals (yfinance where available)")
            fund_df = pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"])
            st.dataframe(fund_df)

            st.subheader("AI-powered plain language insight")
            ai_text = ai_summary_safe(ticker, fundamentals, tech_latest, final_score)
            st.info(ai_text)

def analyze_portfolio(tickers_list, portfolio_data):
    """Function to run the portfolio analysis with P&L calculation."""
    st.subheader("Current Portfolio Status")
    
    # Calculate P&L for each stock
    pnl_data = []
    total_invested = 0
    total_current_value = 0
    
    for _, row in portfolio_data.iterrows():
        ticker = row['Ticker']
        quantity = row['Quantity']
        avg_price = row['Avg. Buying Price']
        
        info = fetch_info(ticker)
        current_price = safe_get(info, "regularMarketPrice")
        
        if current_price is None or pd.isna(current_price):
            pnl_data.append({
                'Ticker': ticker,
                'Quantity': quantity,
                'Avg. Buying Price': avg_price,
                'Current Price': "N/A",
                'Invested Value': quantity * avg_price,
                'Current Value': "N/A",
                'P&L': "N/A",
                'Status': "Data Missing"
            })
            total_invested += quantity * avg_price
            continue

        invested_value = quantity * avg_price
        current_value = quantity * current_price
        pnl = current_value - invested_value
        status = "Profit 😊" if pnl >= 0 else "Loss 😥"
        
        pnl_data.append({
            'Ticker': ticker,
            'Quantity': quantity,
            'Avg. Buying Price': f'₹{avg_price:,.2f}',
            'Current Price': f'₹{current_price:,.2f}',
            'Invested Value': f'₹{invested_value:,.2f}',
            'Current Value': f'₹{current_value:,.2f}',
            'P&L': f'₹{pnl:,.2f}',
            'Status': status
        })
        
        total_invested += invested_value
        total_current_value += current_value

    pnl_df = pd.DataFrame(pnl_data)
    st.dataframe(pnl_df, use_container_width=True, hide_index=True)
    
    # Display overall metrics
    total_pnl = total_current_value - total_invested
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Total Invested Value", f"₹{total_invested:,.2f}")
    col_metric2.metric("Total Current Value", f"₹{total_current_value:,.2f}")
    col_metric3.metric("Overall P&L", f"₹{total_current_value:,.2f}", delta=f"₹{total_pnl:,.2f}")

# ------------------------------
# UI: sidebar for personalization
# ------------------------------
st.sidebar.header("Configuration & Settings")
st.session_state.weight_fund = st.sidebar.slider("Weight to Fundamentals (%)", min_value=0, max_value=100, value=60, step=5) / 100.0
st.session_state.lookback_period = st.sidebar.selectbox("Price lookback for analysis", options=["6mo","1y","2y"], index=1)
st.session_state.market_index = st.sidebar.text_input("Market index for beta (yfinance ticker)", value="^NSEI")
st.session_state.show_tooltips = st.sidebar.checkbox("Show learning tooltips (help text)", value=True)

# ------------------------------
# Main App Body
# ------------------------------

# Navigation buttons
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("Individual Stock Analysis", use_container_width=True):
        st.session_state['analysis_mode'] = 'individual'
with col_nav2:
    if st.button("Portfolio Analysis", use_container_width=True):
        st.session_state['analysis_mode'] = 'portfolio'

st.markdown("---")

if st.session_state['analysis_mode'] == 'individual':
    st.header("Individual Stock Analysis")
    ticker_input = st.text_input("Enter single ticker (e.g., INFY.NS):", value="INFY.NS").upper()
    if st.button("Analyze Stock"):
        if not ticker_input:
            st.error("Enter a valid ticker.")
        else:
            with st.spinner("Analyzing stock..."):
                analyze_individual_stock(ticker_input, st.session_state.weight_fund, st.session_state.show_tooltips)

else: # Portfolio Analysis Mode
    st.header("Portfolio Analysis")
    st.markdown("Edit your portfolio holdings below. Change values by double-clicking a cell.")

    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame({
            'Ticker': ['INFY.NS', 'RELIANCE.NS', 'TCS.NS'],
            'Quantity': [10, 5, 8],
            'Avg. Buying Price': [1400.00, 2500.00, 3200.00]
        })

    edited_df = st.data_editor(
        st.session_state.portfolio_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Enter a valid NSE ticker"),
            "Quantity": st.column_config.NumberColumn("Quantity", format="%d", min_value=1),
            "Avg. Buying Price": st.column_config.NumberColumn("Avg. Buying Price", format="₹%.2f", min_value=0.01)
        },
        num_rows="dynamic",
        use_container_width=True
    )
    
    st.session_state.portfolio_df = edited_df

    if st.button("Analyze My Portfolio"):
        if not edited_df.empty:
            with st.spinner("Analyzing portfolio..."):
                analyze_portfolio(edited_df['Ticker'].tolist(), edited_df)
        else:
            st.warning("Please add some stocks to your portfolio to analyze.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("Notes: This is an MVP. Findings are educational only — not financial advice.")
