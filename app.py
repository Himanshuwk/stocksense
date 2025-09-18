# app.py - StockSense MVP (yfinance + OpenAI for AI summaries)
# Features: personalized scoring, AI insights, sector heatmap (basic), portfolio analysis, learning tooltips.

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
st.title("📈 StockSense — Smart Stock Selection (MVP)")
st.markdown(
    "Personalized stock scoring + AI insights for Indian stocks. "
    "Use NSE tickers like `RELIANCE.NS`, `INFY.NS`, `TCS.NS`."
)

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
    close = df["Close"].dropna()
    result = pd.DataFrame(index=close.index)
    result["Close"] = close

    # RSI
    try:
        result["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    except Exception:
        result["RSI"] = np.nan

    # SMA
    result["SMA50"] = close.rolling(50).mean()
    result["SMA200"] = close.rolling(200).mean()

    # MACD diff
    try:
        result["MACD"] = ta.trend.MACD(close).macd_diff()
    except Exception:
        result["MACD"] = np.nan

    # Bollinger
    try:
        bb = ta.volatility.BollingerBands(close)
        result["BB_high"] = bb.bollinger_hband()
        result["BB_low"] = bb.bollinger_lband()
    except Exception:
        result["BB_high"] = np.nan
        result["BB_low"] = np.nan

    # Volume if present
    if "Volume" in df.columns:
        result["Volume"] = df["Volume"]

    # Daily returns
    result["Return"] = close.pct_change()
    return result

def safe_get(dct, key, default=None):
    v = dct.get(key, default)
    return default if v in (None, "", "None") else v

def calc_beta(series_stock, series_market):
    """Calculate beta of stock vs market using aligned daily returns."""
    # require overlapping indices
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
    # fundamentals: dict with keys Debt/Equity (de_ratio), returnOnEquity (roe), trailingPE, promoter if available
    fund_points = 0
    # debt/equity
    de = fundamentals.get("de_ratio")
    if de is not None:
        try:
            if de < 1:
                fund_points += 20
        except:
            pass
    # ROE
    roe = fundamentals.get("roe")
    if roe is not None:
        try:
            if roe > 0.12:
                fund_points += 20
        except:
            pass
    # P/E check (lower than sector maybe good) - basic
    pe = fundamentals.get("pe")
    if pe is not None and pe > 0:
        if pe < 30:
            fund_points += 20
    # promoter holding placeholder (if available as fraction)
    prom = fundamentals.get("promoter_holding")
    if prom is not None:
        try:
            if prom > 0.5:
                fund_points += 20
        except:
            pass

    # Technical scoring: use RSI and price vs SMA50
    tech_points = 0
    rsi = tech_latest.get("RSI")
    price = tech_latest.get("Close")
    sma50 = tech_latest.get("SMA50")
    if rsi is not None and not np.isnan(rsi):
        if 30 < rsi < 70:
            tech_points += 20
    if price is not None and sma50 is not None and not np.isnan(sma50):
        if price > sma50:
            tech_points += 20

    # Normalize each to 0-100 (here max fund_points=80, tech_points=40) — scale to 0-100:
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
    # Build a concise prompt
    prompt = (
        f"Explain to a beginner investor in 2-3 simple sentences the situation for {ticker}.\n"
        f"Fundamentals: {fundamentals}\n"
        f"Technicals: RSI={tech_latest.get('RSI')}, Close={tech_latest.get('Close')}, SMA50={tech_latest.get('SMA50')}\n"
        f"Overall score: {final_score} out of 100.\n"
        "Keep it short, no jargon. At the end, give one sentence recommendation: Strong Buy / Watchlist / Avoid.\n"
        "Include a 1-line explanation why."
    )
    if not OPENAI_KEY:
        # fallback simple summary
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        txt = response.choices[0].message.content.strip()
        return txt
    except Exception as e:
        return f"AI unavailable: {e}"

# ------------------------------
# UI: sidebar for personalization & portfolio input
# ------------------------------
st.sidebar.header("Configuration & Portfolio")

weight_fund = st.sidebar.slider("Weight to Fundamentals (%)", min_value=0, max_value=100, value=60, step=5)
weight_fund = weight_fund / 100.0  # convert to 0-1

st.sidebar.markdown("**Portfolio input** (comma separated tickers). Example: `INFY.NS, RELIANCE.NS`")
portfolio_input = st.sidebar.text_input("Portfolio tickers:", value="INFY.NS, RELIANCE.NS")
tickers_list = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]

lookback_period = st.sidebar.selectbox("Price lookback for analysis", options=["6mo","1y","2y"], index=1)
market_index = st.sidebar.text_input("Market index for beta (yfinance ticker)", value="^NSEI")

st.sidebar.markdown("---")
st.sidebar.markdown("**Learning mode (tooltips)**")
if st.sidebar.checkbox("Show learning tooltips (help text)", value=True):
    show_tooltips = True
else:
    show_tooltips = False

# ------------------------------
# Main UI: single stock analysis + portfolio
# ------------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.header("Single stock quick analyze")
    ticker = st.text_input("Enter single ticker (e.g., INFY.NS):", value="INFY.NS").upper()
    if st.button("Analyze this stock"):
        if not ticker:
            st.error("Enter a valid ticker.")
        else:
            # Fetch
            df = fetch_price_data(ticker, period=lookback_period)
            info = fetch_info(ticker)
            if df is None:
                st.error("No price data found for ticker. Check format (e.g., INFY.NS).")
            else:
                tech = compute_technical_indicators(df)
                tech_latest = {
                    "Close": float(tech["Close"].iloc[-1]) if not tech["Close"].isna().all() else None,
                    "RSI": float(tech["RSI"].iloc[-1]) if "RSI" in tech.columns else None,
                    "SMA50": float(tech["SMA50"].iloc[-1]) if "SMA50" in tech.columns else None,
                    "SMA200": float(tech["SMA200"].iloc[-1]) if "SMA200" in tech.columns else None,
                }

                # Fundamentals from yfinance (best-effort)
                fundamentals = {
                    "pe": safe_get(info, "trailingPE", None) or safe_get(info, "forwardPE", None),
                    "roe": safe_get(info, "returnOnEquity", None),
                    "de_ratio": safe_get(info, "debtToEquity", None),
                    "promoter_holding": None  # placeholder; yfinance rarely has promoter
                }
                # sector
                sector = safe_get(info, "sector", "Unknown")

                # Score
                final_score, details = score_stock(fundamentals, tech_latest, weight_fund)
                label = "🟢 Strong Buy" if final_score>=80 else ("🟡 Watchlist" if final_score>=50 else "🔴 Avoid")

                # Show results
                st.subheader(f"{ticker} — {sector}")
                st.metric("StockSense score", f"{final_score}/100", delta=label)

                if show_tooltips:
                    st.caption("Score = weighted blend of fundamentals & technicals. Adjust weight in the sidebar.")

                # Plots: price with SMA & Bollinger
                fig_df = pd.DataFrame({
                    "Close": tech["Close"],
                    "SMA50": tech["SMA50"],
                    "SMA200": tech["SMA200"],
                    "BB_high": tech.get("BB_high"),
                    "BB_low": tech.get("BB_low")
                })
                fig = px.line(fig_df.reset_index(), x=fig_df.index, y=fig_df.columns,
                              labels={"value":"Price", "index":"Date"}, title=f"{ticker} Price & Indicators")
                st.plotly_chart(fig, use_container_width=True)

                # RSI & MACD
                st.subheader("RSI & MACD")
                rsi_fig = px.line(tech.reset_index(), x=tech.index, y="RSI", title="RSI")
                macd_fig = px.line(tech.reset_index(), x=tech.index, y="MACD", title="MACD")
                st.plotly_chart(rsi_fig, use_container_width=True)
                st.plotly_chart(macd_fig, use_container_width=True)

                # Fundamentals table
                st.subheader("Key fundamentals (yfinance where available)")
                fund_df = pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"])
                st.dataframe(fund_df)

                # AI insights
                st.subheader("AI-powered plain language insight")
                ai_text = ai_summary_safe(ticker, fundamentals, tech_latest, final_score)
                st.info(ai_text)

with col2:
    st.header("Portfolio analysis (quick)")
    if not tickers_list:
        st.info("Add tickers in the sidebar (comma separated).")
    else:
        # gather data
        portfolio_data = {}
        failed = []
        for tk in tickers_list:
            df = fetch_price_data(tk, period=lookback_period)
            if df is None:
                failed.append(tk)
                continue
            portfolio_data[tk] = {
                "price_df": df,
                "info": fetch_info(tk),
                "tech": compute_technical_indicators(df)
            }

        if failed:
            st.warning(f"No price data for: {', '.join(failed)}. Remove or check tickers.")

        if portfolio_data:
            # Sector distribution
            sectors = {}
            for tk, obj in portfolio_data.items():
                s = safe_get(obj["info"], "sector", "Unknown")
                sectors.setdefault(s, []).append(tk)
            sector_counts = {k: len(v) for k, v in sectors.items()}
            sec_df = pd.DataFrame({"sector": list(sector_counts.keys()), "count": list(sector_counts.values())})
            st.subheader("Portfolio sector distribution")
            fig = px.pie(sec_df, values="count", names="sector", title="Sectors in Portfolio")
            st.plotly_chart(fig, use_container_width=True)

            # Portfolio returns & volatility
            returns_df = pd.DataFrame()
            for tk, obj in portfolio_data.items():
                r = obj["tech"]["Return"].rename(tk)
                returns_df = pd.concat([returns_df, r], axis=1)
            returns_df = returns_df.dropna(how="all")

            if returns_df.shape[0] < 10:
                st.warning("Not enough history to compute portfolio metrics.")
            else:
                # equal weight portfolio
                weights = np.array([1/len(returns_df.columns)]*len(returns_df.columns))
                daily_portfolio = returns_df.fillna(0).dot(weights)
                ann_return = ((1 + daily_portfolio.mean())**252 - 1) * 100
                ann_vol = daily_portfolio.std() * np.sqrt(252) * 100
                st.metric("Estimated annual return (%)", f"{ann_return:.2f}")
                st.metric("Estimated annual volatility (%)", f"{ann_vol:.2f}")

                # Beta per stock vs market index
                mdf = fetch_price_data(market_index, period=lookback_period)
                if mdf is not None:
                    market_returns = mdf["Close"].pct_change()
                    betas = {}
                    for tk, obj in portfolio_data.items():
                        stock_returns = obj["tech"]["Return"]
                        beta = calc_beta(stock_returns, market_returns)
                        betas[tk] = round(beta, 3) if beta is not None else None
                    beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=["Beta"])
                    st.subheader("Beta vs market")
                    st.dataframe(beta_df)
                else:
                    st.info("Market index data not available for beta calculation.")

# ------------------------------
# Footer / Next steps
# ------------------------------
st.markdown("---")
st.write("Notes: This is an MVP. Findings are educational only — not financial advice.")
st.write("Next improvements: Alerts (email/telegram), backtesting, news sentiment, better fundamentals source (NSE/Tickertape).")
