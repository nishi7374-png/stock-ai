"""
株式テクニカル分析 × Claude AI判断 - Streamlitアプリ（履歴保存機能付き）
"""

import os
import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from anthropic import Anthropic

# ─── 履歴ファイルのパス ───────────────────────────────────────────
HISTORY_FILE = Path("analysis_history.json")

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def add_to_history(ticker, company_name, ind, result_text):
    history = load_history()
    history.insert(0, {
        "date": pd.Timestamp.now().strftime("%Y/%m/%d %H:%M"),
        "ticker": ticker,
        "company_name": company_name,
        "price": ind["price"],
        "change_pct": ind["change_pct"],
        "rsi": ind["rsi"],
        "result": result_text,
    })
    history = history[:30]  # 最新30件を保持
    save_history(history)

# ─── テクニカル指標の計算 ─────────────────────────────────────────
def calc_ma(close, window):
    return close.rolling(window).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(close, window=20, num_std=2):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str = "6mo"):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                info = yf.Ticker(ticker).info
                name = info.get("longName") or info.get("shortName") or ticker
                return df, name
        except Exception:
            pass
        time.sleep(1)
    return None, ticker

def build_indicators(df):
    close = df["Close"].squeeze()
    ma5, ma25, ma75 = calc_ma(close, 5), calc_ma(close, 25), calc_ma(close, 75)
    rsi = calc_rsi(close)
    macd_line, signal_line, histogram = calc_macd(close)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)

    latest = {
        "price":      float(close.iloc[-1]),
        "prev_price": float(close.iloc[-2]),
        "ma5":        float(ma5.iloc[-1]),
        "ma25":       float(ma25.iloc[-1]),
        "ma75":       float(ma75.iloc[-1]) if len(close) >= 75 else float("nan"),
        "rsi":        float(rsi.iloc[-1]),
        "macd":       float(macd_line.iloc[-1]),
        "macd_sig":   float(signal_line.iloc[-1]),
        "macd_hist":  float(histogram.iloc[-1]),
        "macd_prev":  float(histogram.iloc[-2]),
        "volume":     float(df["Volume"].squeeze().iloc[-1]),
        "volume_ma5": float(df["Volume"].squeeze().rolling(5).mean().iloc[-1]),
        "bb_upper":   float(bb_upper.iloc[-1]),
        "bb_mid":     float(bb_mid.iloc[-1]),
        "bb_lower":   float(bb_lower.iloc[-1]),
    }
    latest["change_pct"] = (latest["price"] - latest["prev_price"]) / latest["prev_price"] * 100

    series = {
        "close": close, "ma5": ma5, "ma25": ma25, "ma75": ma75,
        "rsi": rsi, "macd_line": macd_line, "signal_line": signal_line,
        "histogram": histogram, "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
    }
    return latest, series

# ─── チャート描画 ─────────────────────────────────────────────────
def draw_chart(df, series, ticker, company_name):
    close = series["close"]
    dates = close.index
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=df["Close"].squeeze(),
        name="株価",
        increasing_line_color="#4ade80",
        decreasing_line_color="#f87171",
    ))
    for label, col, color in [("MA5", "ma5", "#60a5fa"), ("MA25", "ma25", "#fbbf24"), ("MA75", "ma75", "#c084fc")]:
        fig.add_trace(go.Scatter(x=dates, y=series[col], name=label, line=dict(color=color, width=1.2)))
    fig.add_trace(go.Scatter(x=dates, y=series["bb_upper"], name="BB上限",
                             line=dict(color="#94a3b8", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=dates, y=series["bb_lower"], name="BB下限",
                             line=dict(color="#94a3b8", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(148,163,184,0.05)"))
    fig.update_layout(
        title=f"{ticker}（{company_name}） 株価チャート",
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickformat="%m/%d", dtick="M1"),
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

def draw_rsi(series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["rsi"].index, y=series["rsi"],
                             name="RSI", line=dict(color="#60a5fa", width=2)))
    fig.add_hline(y=70, line_dash="dot", line_color="#f87171", annotation_text="買われすぎ70")
    fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", annotation_text="売られすぎ30")
    fig.update_layout(
        template="plotly_dark",
        height=180,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 100]),
        xaxis=dict(tickformat="%m/%d"),
    )
    return fig

def draw_macd(series):
    fig = go.Figure()
    colors = ["#4ade80" if v >= 0 else "#f87171" for v in series["histogram"]]
    fig.add_trace(go.Bar(x=series["histogram"].index, y=series["histogram"],
                         name="ヒストグラム", marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=series["macd_line"].index, y=series["macd_line"],
                             name="MACD", line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=series["signal_line"].index, y=series["signal_line"],
                             name="Signal", line=dict(color="#fbbf24", width=1.5)))
    fig.update_layout(
        template="plotly_dark",
        height=180,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickformat="%m/%d"),
    )
    return fig

def draw_volume(df):
    colors = ["#4ade80" if c >= o else "#f87171"
              for c, o in zip(df["Close"].squeeze(), df["Open"].squeeze())]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"].squeeze(),
        name="出来高",
        marker_color=colors,
        opacity=0.8,
    ))
    fig.update_layout(
        template="plotly_dark",
        height=180,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="出来高"),
        xaxis=dict(tickformat="%m/%d"),
    )
    return fig

# ─── Claude AI分析 ────────────────────────────────────────────────
def ask_claude_stream(client, ticker, company_name, ind):
    prompt = f"""あなたは株式テクニカルアナリストです。以下の指標をもとに、プロの視点で売買判断を日本語で述べてください。

【銘柄】{ticker}（{company_name}）
【現在値】{ind['price']:.2f}（前日比 {ind['change_pct']:+.2f}%）
【移動平均線】MA5={ind['ma5']:.2f} / MA25={ind['ma25']:.2f} / MA75={ind['ma75']:.2f}
【RSI(14)】{ind['rsi']:.1f}
【MACD】ライン={ind['macd']:.3f} / シグナル={ind['macd_sig']:.3f} / ヒスト={ind['macd_hist']:.3f}（前日={ind['macd_prev']:.3f}）
【ボリンジャーバンド】上限={ind['bb_upper']:.2f} / 中央={ind['bb_mid']:.2f} / 下限={ind['bb_lower']:.2f}
【出来高】直近={ind['volume']:.0f} / 5日平均={ind['volume_ma5']:.0f}

以下の構成で回答してください：
1. **総合判断**：「🟢 買い」「🔴 売り」「⚪ 様子見」のいずれかと確信度（高/中/低）
2. **根拠**：各指標が示すシグナルの解説（箇条書き）
3. **注目ポイント**：特に重要な指標や水準
4. **アドバイス**：具体的な行動提案（エントリー・利確・損切りの目安など）
5. **リスク**：注意すべきリスク要因

※ 投資判断はあくまで参考情報です。"""

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text

# ─── ページ設定 ───────────────────────────────────────────────────
st.set_page_config(page_title="株式AI分析", page_icon="📈", layout="wide")

# ─── サイドバー ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")
    api_key = st.text_input(
        "Anthropic APIキー",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="https://console.anthropic.com で取得できます",
    )
    period = st.selectbox("取得期間", ["3mo", "6mo", "1y", "2y"], index=1)
    st.markdown("---")
    st.markdown("**銘柄コード例**")
    st.markdown("🇯🇵 トヨタ: `7203.T`")
    st.markdown("🇯🇵 ソフトバンク: `9984.T`")
    st.markdown("🇺🇸 Apple: `AAPL`")
    st.markdown("🇺🇸 NVIDIA: `NVDA`")
    st.markdown("---")

    # 分析履歴
    st.markdown("**📋 分析履歴**")
    history = load_history()
    if not history:
        st.caption("まだ履歴がありません")
    else:
        # 履歴の全削除ボタン
        if st.button("🗑️ 履歴を全削除", use_container_width=True):
            save_history([])
            st.rerun()

        for i, h in enumerate(history):
            change_emoji = "🟢" if h["change_pct"] >= 0 else "🔴"
            company = h.get("company_name", h["ticker"])
            label = f"{h['date']}  {h['ticker']}（{company}）  {change_emoji}"
            with st.expander(label):
                st.caption(f"価格: {h['price']:,.2f}　前日比: {h['change_pct']:+.2f}%　RSI: {h['rsi']:.1f}")
                st.markdown(h["result"])

    st.markdown("---")
    st.caption("⚠️ 投資は自己責任で。このツールは参考情報です。")

# ─── メイン ───────────────────────────────────────────────────────
st.title("📈 株式テクニカル分析 × Claude AI")
st.caption("リアルタイム株価データ × AIによる売買判断")

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.text_input("銘柄コードを入力", placeholder="例: 7203.T / AAPL / 9984.T", label_visibility="collapsed")
with col_btn:
    analyze_btn = st.button("🔍 分析する", use_container_width=True, type="primary")

if analyze_btn and ticker_input:
    ticker = ticker_input.strip().upper()

    if not api_key:
        st.error("サイドバーにAnthropicのAPIキーを入力してください。")
        st.stop()

    with st.spinner(f"{ticker} のデータを取得中…"):
        df, company_name = fetch_data(ticker, period)

    if df is None or df.empty:
        st.error(f"「{ticker}」のデータが取得できませんでした。\n\n"
                 "・銘柄コードを確認してください（日本株は末尾に `.T` が必要です）\n"
                 "・時間をおいて再度お試しください")
        st.stop()

    ind, series = build_indicators(df)

    st.markdown("---")
    change_color = "normal" if ind["change_pct"] >= 0 else "inverse"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("現在値", f"{ind['price']:,.2f}", f"{ind['change_pct']:+.2f}%", delta_color=change_color)
    c2.metric("MA5",  f"{ind['ma5']:,.2f}")
    c3.metric("MA25", f"{ind['ma25']:,.2f}")
    c4.metric("RSI",  f"{ind['rsi']:.1f}")
    c5.metric("MACDヒスト", f"{ind['macd_hist']:.3f}")

    st.plotly_chart(draw_chart(df, series, ticker, company_name), use_container_width=True)

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.caption("RSI (14日)")
        st.plotly_chart(draw_rsi(series), use_container_width=True)
    with col_macd:
        st.caption("MACD (12-26-9)")
        st.plotly_chart(draw_macd(series), use_container_width=True)

    st.caption("出来高")
    st.plotly_chart(draw_volume(df), use_container_width=True)

    st.markdown("---")
    st.subheader(f"🤖 テクニカル分析レポート：{ticker}（{company_name}）")

    client = Anthropic(api_key=api_key)
    response_box = st.empty()
    full_text = ""
    with st.spinner("Claude が分析中…"):
        for chunk in ask_claude_stream(client, ticker, company_name, ind):
            full_text += chunk
            response_box.markdown(full_text + "▌")
    response_box.markdown(full_text)

    # 履歴に保存
    add_to_history(ticker, company_name, ind, full_text)
    st.success("✅ 分析結果を履歴に保存しました（サイドバーで確認できます）")

elif analyze_btn and not ticker_input:
    st.warning("銘柄コードを入力してください。")
