"""
株式テクニカル分析 × Claude AI判断 - Streamlitアプリ
（スクリーニング・アラート・前日比較機能付き）
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
import requests

# ─── ファイルパス ─────────────────────────────────────────────────
HISTORY_FILE    = Path("analysis_history.json")
WATCHLIST_FILE  = Path("watchlist.json")
SCAN_TODAY_FILE = Path("scan_today.json")
SCAN_PREV_FILE  = Path("scan_prev.json")

# ─── アラート設定 ─────────────────────────────────────────────────
ALERT_SCORE     = 6   # このスコア以上をアラート対象
ALERT_TOP_N     = 3   # アラートで表示する件数
COMPARE_TOP_N   = 5   # 前日比較で表示する件数

# ─── 履歴 ─────────────────────────────────────────────────────────
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def add_to_history(ticker, company_name, ind, result_text):
    history = load_history()
    history.insert(0, {
        "date":         pd.Timestamp.now().strftime("%Y/%m/%d %H:%M"),
        "ticker":       ticker,
        "company_name": company_name,
        "price":        ind["price"],
        "change_pct":   ind["change_pct"],
        "rsi":          ind["rsi"],
        "result":       result_text,
    })
    save_history(history[:30])

# ─── ウォッチリスト ───────────────────────────────────────────────
def load_watchlist():
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    try:
        cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
        return [{"ticker": t, "label": t} for t in cfg.get("tickers", [])]
    except Exception:
        return []

def save_watchlist(watchlist):
    WATCHLIST_FILE.write_text(json.dumps(watchlist, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo  = st.secrets.get("GITHUB_REPO", "")
        if not token or not repo:
            return
        url = f"https://api.github.com/repos/{repo}/contents/watchlist.json"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        r = requests.get(url, headers=headers)
        sha = r.json().get("sha", "") if r.status_code == 200 else ""
        content = json.dumps(watchlist, ensure_ascii=False, indent=2)
        import base64
        payload = {
            "message": "update watchlist",
            "content": base64.b64encode(content.encode()).decode(),
            "sha": sha
        }
        requests.put(url, headers=headers, json=payload)
    except Exception:
        pass

# ─── スキャン結果の読み込み ───────────────────────────────────────
def load_scan(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

# ─── アラートチェック ─────────────────────────────────────────────
def get_alert_stocks():
    """スコアがALERT_SCORE以上の上位ALERT_TOP_N銘柄を返す"""
    scan = load_scan(SCAN_TODAY_FILE)
    if not scan:
        return [], None
    hits = [r for r in scan["results"] if r["score"] >= ALERT_SCORE]
    return hits[:ALERT_TOP_N], scan["scanned_at"]

# ─── 前日比較 ─────────────────────────────────────────────────────
def get_score_changes():
    """スコアが前日から最も大きく上昇した上位COMPARE_TOP_N銘柄を返す"""
    today = load_scan(SCAN_TODAY_FILE)
    prev  = load_scan(SCAN_PREV_FILE)
    if not today or not prev:
        return [], None, None

    prev_map = {r["ticker"]: r["score"] for r in prev["results"]}
    changes = []
    for r in today["results"]:
        prev_score = prev_map.get(r["ticker"])
        if prev_score is not None:
            diff = r["score"] - prev_score
            if diff > 0:  # 上昇した銘柄だけ
                changes.append({**r, "prev_score": prev_score, "diff": diff})

    changes.sort(key=lambda x: x["diff"], reverse=True)
    return changes[:COMPARE_TOP_N], today["scanned_at"], prev["scanned_at"]

# ─── テクニカル指標の計算 ─────────────────────────────────────────
def calc_ma(close, window):
    return close.rolling(window).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(close, window=20, num_std=2):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def calc_atr(df, period=14):
    high       = df["High"].squeeze()
    low        = df["Low"].squeeze()
    close_prev = df["Close"].squeeze().shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_stochastics(df, k_period=14, d_period=3):
    high         = df["High"].squeeze()
    low          = df["Low"].squeeze()
    close        = df["Close"].squeeze()
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    d = k.rolling(d_period).mean()
    return k, d

@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str = "6mo"):
    last_error = None
    for attempt in range(3):
        try:
            tk  = yf.Ticker(ticker)
            df  = tk.history(period=period, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"必要なカラムがありません: {required_cols - set(df.columns)}")
            if df is None or df.empty or len(df) < 20:
                raise ValueError(f"データが少なすぎます（{len(df) if df is not None else 0}件）")
            name = ticker
            try:
                info = tk.info
                name = info.get("longName") or info.get("shortName") or ticker
            except Exception:
                pass
            return df, name
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    st.error(f"データ取得に失敗しました（最終エラー: {last_error}）")
    return None, ticker

def build_indicators(df):
    close = df["Close"].squeeze()
    ma5, ma25, ma75 = calc_ma(close, 5), calc_ma(close, 25), calc_ma(close, 75)
    rsi = calc_rsi(close)
    macd_line, signal_line, histogram = calc_macd(close)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    atr    = calc_atr(df)
    stoch_k, stoch_d = calc_stochastics(df)

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
        "atr":        float(atr.iloc[-1]),
        "stoch_k":    float(stoch_k.iloc[-1]),
        "stoch_d":    float(stoch_d.iloc[-1]),
    }
    latest["change_pct"]      = (latest["price"] - latest["prev_price"]) / latest["prev_price"] * 100
    latest["stop_loss_buy"]   = latest["price"] - 1.5 * latest["atr"]
    latest["stop_loss_sell"]  = latest["price"] + 1.5 * latest["atr"]

    series = {
        "close": close, "ma5": ma5, "ma25": ma25, "ma75": ma75,
        "rsi": rsi, "macd_line": macd_line, "signal_line": signal_line,
        "histogram": histogram, "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "atr": atr, "stoch_k": stoch_k, "stoch_d": stoch_d,
    }
    return latest, series

# ─── スクリーニング（手動実行用） ────────────────────────────────
def score_signals(ind):
    signals = []
    score   = 0
    if 30 <= ind["rsi"] <= 50:
        score += 1; signals.append(("✅", "RSI回復中",          f"RSI={ind['rsi']:.1f}"))
    elif ind["rsi"] < 30:
        score += 0.5; signals.append(("⚠️", "RSI売られすぎ",   f"RSI={ind['rsi']:.1f}"))
    if ind["macd_hist"] > ind["macd_prev"]:
        score += 1; signals.append(("✅", "MACD改善",           f"ヒスト={ind['macd_hist']:.3f}"))
    if ind["macd"] > ind["macd_sig"]:
        score += 1; signals.append(("✅", "MACDゴールデンクロス", ""))
    if ind["price"] > ind["ma25"]:
        score += 1; signals.append(("✅", "MA25上抜け",         f"{ind['price']:.2f}>{ind['ma25']:.2f}"))
    if ind["ma5"] > ind["ma25"]:
        score += 1; signals.append(("✅", "短期MA上昇",         ""))
    if ind["stoch_k"] > ind["stoch_d"] and 20 <= ind["stoch_k"] <= 80:
        score += 1; signals.append(("✅", "ストキャス上昇",     f"%K={ind['stoch_k']:.1f}"))
    if ind["volume"] > ind["volume_ma5"]:
        score += 1; signals.append(("✅", "出来高増加",         ""))
    bb_range = ind["bb_upper"] - ind["bb_lower"]
    if bb_range > 0 and (ind["price"] - ind["bb_lower"]) / bb_range < 0.3:
        score += 1; signals.append(("✅", "BB下限付近",         ""))
    return score, signals

def run_screening(watchlist, period="6mo"):
    results  = []
    progress = st.progress(0, text="スキャン中...")
    total    = len(watchlist)
    for i, item in enumerate(watchlist):
        ticker = item["ticker"]
        label  = item.get("label") or ticker
        progress.progress((i + 1) / total, text=f"スキャン中… {label}（{ticker}）")
        df, company_name = fetch_data(ticker, period)
        if df is None or df.empty:
            results.append({"ticker": ticker, "label": label, "company_name": company_name,
                            "score": -1, "signals": [], "ind": None, "error": True})
            continue
        ind, _ = build_indicators(df)
        score, signals = score_signals(ind)
        results.append({"ticker": ticker, "label": label, "company_name": company_name,
                        "score": score, "signals": signals, "ind": ind, "error": False})
    progress.empty()
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def ask_claude_screening(client, candidates):
    lines = []
    for r in candidates:
        ind      = r["ind"]
        sig_text = "、".join([s[1] for s in r["signals"]])
        lines.append(
            f"・{r['ticker']}（{r['company_name']}）スコア{r['score']}/8　"
            f"現在値={ind['price']:,.2f}（{ind['change_pct']:+.2f}%）　"
            f"RSI={ind['rsi']:.1f}　MACD_hist={ind['macd_hist']:.3f}　シグナル: {sig_text}"
        )
    prompt = f"""あなたは株式テクニカルアナリストです。
以下はテクニカルスクリーニングで選ばれた買い候補銘柄の一覧です（スコア高い順）。
各銘柄を比較し、特に注目すべき銘柄とその理由を日本語で解説してください。

【候補銘柄】
{chr(10).join(lines)}

以下の構成で回答してください：
1. **総評**：全体的な市場環境のコメント
2. **最注目銘柄**：1〜2銘柄を選んで理由を詳しく
3. **各銘柄の簡評**：それぞれ1〜2文で
4. **注意点**：スクリーニング結果を使う上でのリスク

※投資判断はあくまで参考情報です。"""
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text

# ─── チャート描画 ─────────────────────────────────────────────────
def draw_chart(df, series, ticker, company_name):
    close = series["close"]
    dates = close.index
    fig   = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=df["Close"].squeeze(), name="株価",
        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
    ))
    for label, col, color in [("MA5","ma5","#60a5fa"),("MA25","ma25","#fbbf24"),("MA75","ma75","#c084fc")]:
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
        template="plotly_dark", height=420,
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
    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=10,r=10,t=20,b=10),
                      yaxis=dict(range=[0,100]), xaxis=dict(tickformat="%m/%d"))
    return fig

def draw_macd(series):
    fig    = go.Figure()
    colors = ["#4ade80" if v >= 0 else "#f87171" for v in series["histogram"]]
    fig.add_trace(go.Bar(x=series["histogram"].index, y=series["histogram"],
                         name="ヒストグラム", marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=series["macd_line"].index, y=series["macd_line"],
                             name="MACD", line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=series["signal_line"].index, y=series["signal_line"],
                             name="Signal", line=dict(color="#fbbf24", width=1.5)))
    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=10,r=10,t=20,b=10),
                      xaxis=dict(tickformat="%m/%d"))
    return fig

def draw_volume(df):
    colors = ["#4ade80" if c >= o else "#f87171"
              for c, o in zip(df["Close"].squeeze(), df["Open"].squeeze())]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(),
                         name="出来高", marker_color=colors, opacity=0.8))
    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=10,r=10,t=20,b=10),
                      yaxis=dict(title="出来高"), xaxis=dict(tickformat="%m/%d"))
    return fig

def draw_stochastics(series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["stoch_k"].index, y=series["stoch_k"],
                             name="%K", line=dict(color="#60a5fa", width=2)))
    fig.add_trace(go.Scatter(x=series["stoch_d"].index, y=series["stoch_d"],
                             name="%D（シグナル）", line=dict(color="#fbbf24", width=1.5, dash="dash")))
    fig.add_hline(y=80, line_dash="dot", line_color="#f87171", annotation_text="買われすぎ80")
    fig.add_hline(y=20, line_dash="dot", line_color="#4ade80", annotation_text="売られすぎ20")
    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=10,r=10,t=20,b=10),
                      yaxis=dict(range=[0,100]), xaxis=dict(tickformat="%m/%d"))
    return fig

def draw_atr(series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["atr"].index, y=series["atr"],
                             name="ATR(14)", line=dict(color="#a78bfa", width=2),
                             fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"))
    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=10,r=10,t=20,b=10),
                      xaxis=dict(tickformat="%m/%d"), yaxis=dict(title="値幅"))
    return fig

# ─── Claude AI分析（個別） ────────────────────────────────────────
def ask_claude_stream(client, ticker, company_name, ind):
    prompt = f"""あなたは株式テクニカルアナリストです。以下の指標をもとに、プロの視点で売買判断を日本語で述べてください。

【銘柄】{ticker}（{company_name}）
【現在値】{ind['price']:.2f}（前日比 {ind['change_pct']:+.2f}%）
【移動平均線】MA5={ind['ma5']:.2f} / MA25={ind['ma25']:.2f} / MA75={ind['ma75']:.2f}
【RSI(14)】{ind['rsi']:.1f}
【MACD】ライン={ind['macd']:.3f} / シグナル={ind['macd_sig']:.3f} / ヒスト={ind['macd_hist']:.3f}（前日={ind['macd_prev']:.3f}）
【ボリンジャーバンド】上限={ind['bb_upper']:.2f} / 中央={ind['bb_mid']:.2f} / 下限={ind['bb_lower']:.2f}
【出来高】直近={ind['volume']:.0f} / 5日平均={ind['volume_ma5']:.0f}
【ATR(14)】{ind['atr']:.2f}　損切り目安：買い={ind['stop_loss_buy']:.2f} / 売り={ind['stop_loss_sell']:.2f}
【ストキャスティクス(14,3)】%K={ind['stoch_k']:.1f} / %D={ind['stoch_d']:.1f}

以下の構成で回答してください：
1. **総合判断**：「🟢 買い」「🔴 売り」「⚪ 様子見」のいずれかと確信度（高/中/低）
2. **根拠**：各指標が示すシグナルの解説（箇条書き）
3. **注目ポイント**：特に重要な指標や水準
4. **アドバイス**：具体的な行動提案（エントリー・利確・ATRを使った損切りの目安など）
5. **リスク**：注意すべきリスク要因

※ 投資判断はあくまで参考情報です。"""

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text

# ══════════════════════════════════════════════════════════════════
# ページ設定
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="株式AI分析", page_icon="📈", layout="wide")

# ─── サイドバー ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")
    api_key = st.text_input(
        "Anthropic APIキー", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="https://console.anthropic.com で取得できます",
    )
    period = st.selectbox("取得期間", ["3mo", "6mo", "1y", "2y"], index=1)
    st.markdown("---")

    st.markdown("**⭐ 注目リスト**")
    watchlist = load_watchlist()
    if not watchlist:
        st.caption("まだ銘柄が登録されていません")
    else:
        for i, item in enumerate(watchlist):
            col_btn_w, col_del = st.columns([4, 1])
            with col_btn_w:
                label = item.get("label") or item["ticker"]
                if st.button(f"📊 {label}　({item['ticker']})", key=f"wl_{i}", use_container_width=True):
                    st.session_state["watchlist_trigger"] = item["ticker"]
            with col_del:
                if st.button("✕", key=f"del_{i}", help="リストから削除"):
                    watchlist.pop(i)
                    save_watchlist(watchlist)
                    st.rerun()

    with st.expander("＋ 銘柄を追加"):
        new_ticker = st.text_input("銘柄コード", placeholder="例: 7203.T / AAPL", key="wl_new_ticker")
        new_label  = st.text_input("表示名（省略可）", placeholder="例: トヨタ", key="wl_new_label")
        if st.button("追加する", key="wl_add", use_container_width=True):
            t = new_ticker.strip().upper()
            if t:
                if any(w["ticker"] == t for w in watchlist):
                    st.warning("すでに登録されています")
                elif len(watchlist) >= 20:
                    st.warning("登録できるのは最大20銘柄です")
                else:
                    watchlist.append({"ticker": t, "label": new_label.strip() or t})
                    save_watchlist(watchlist)
                    st.success(f"{t} を追加しました")
                    st.rerun()
            else:
                st.warning("銘柄コードを入力してください")

    st.markdown("---")
    st.markdown("**銘柄コード例**")
    st.markdown("🇯🇵 トヨタ: `7203.T`")
    st.markdown("🇯🇵 ソフトバンク: `9984.T`")
    st.markdown("🇺🇸 Apple: `AAPL`")
    st.markdown("---")

    st.markdown("**📋 分析履歴**")
    history = load_history()
    if not history:
        st.caption("まだ履歴がありません")
    else:
        if st.button("🗑️ 履歴を全削除", use_container_width=True):
            save_history([])
            st.rerun()
        for h in history:
            change_emoji = "🟢" if h["change_pct"] >= 0 else "🔴"
            company = h.get("company_name", h["ticker"])
            with st.expander(f"{h['date']}  {h['ticker']}（{company}）  {change_emoji}"):
                st.caption(f"価格: {h['price']:,.2f}　前日比: {h['change_pct']:+.2f}%　RSI: {h['rsi']:.1f}")
                st.markdown(h["result"])

    st.markdown("---")
    st.caption("⚠️ 投資は自己責任で。このツールは参考情報です。")

# ══════════════════════════════════════════════════════════════════
# 🔔 アラートバナー（ページ最上部）
# ══════════════════════════════════════════════════════════════════
st.title("📈 株式テクニカル分析 × Claude AI")

alert_stocks, scanned_at = get_alert_stocks()
if alert_stocks:
    names = "　".join([f"**{r['ticker']}**（{r['score']}点）" for r in alert_stocks])
    st.error(
        f"🔔 **本日のアラート** ─ スコア{ALERT_SCORE}点以上が {len(alert_stocks)} 銘柄あります！\n\n"
        f"{names}\n\n"
        f"スキャン時刻: {scanned_at}　／　詳細は「🎯 スクリーニング」タブへ",
        icon="🚨",
    )
elif SCAN_TODAY_FILE.exists():
    st.info("✅ 本日のスキャン済み。アラート対象銘柄はありませんでした。", icon="📭")
else:
    st.warning("⏳ 本日のスキャンデータがまだありません。GitHub Actions が毎朝5時に自動実行します。", icon="⏰")

st.caption("リアルタイム株価データ × AIによる売買判断")

# ─── タブ ────────────────────────────────────────────────────────
tab_single, tab_scan = st.tabs(["🔍 個別分析", "🎯 スクリーニング"])

# ══════════════════════════════════════════════════════════════════
# タブ①：個別分析
# ══════════════════════════════════════════════════════════════════
with tab_single:
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input("銘柄コードを入力",
                                     placeholder="例: 7203.T / AAPL / 9984.T",
                                     label_visibility="collapsed", key="single_ticker")
    with col_btn:
        analyze_btn = st.button("🔍 分析する", use_container_width=True, type="primary", key="single_btn")

    if "watchlist_trigger" in st.session_state and st.session_state["watchlist_trigger"]:
        ticker_input = st.session_state.pop("watchlist_trigger")
        analyze_btn  = True

    if analyze_btn and ticker_input:
        ticker = ticker_input.strip().upper()
        if not api_key:
            st.error("サイドバーにAnthropicのAPIキーを入力してください。")
            st.stop()

        with st.spinner(f"{ticker} のデータを取得中…"):
            df, company_name = fetch_data(ticker, period)

        if df is None or df.empty:
            st.error(
                f"「{ticker}」のデータが取得できませんでした。\n\n"
                "銘柄コードを確認してください（日本株は末尾に `.T` が必要です）"
            )
            st.stop()

        ind, series = build_indicators(df)
        st.markdown("---")
        change_color = "normal" if ind["change_pct"] >= 0 else "inverse"

        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        c1.metric("現在値",       f"{ind['price']:,.2f}",   f"{ind['change_pct']:+.2f}%", delta_color=change_color)
        c2.metric("MA5",          f"{ind['ma5']:,.2f}")
        c3.metric("MA25",         f"{ind['ma25']:,.2f}")
        c4.metric("RSI",          f"{ind['rsi']:.1f}")
        c5.metric("MACDヒスト",   f"{ind['macd_hist']:.3f}")
        c6.metric("ATR(14)",      f"{ind['atr']:.2f}")
        c7.metric("ストキャス%K", f"{ind['stoch_k']:.1f}", f"%D {ind['stoch_d']:.1f}")

        st.plotly_chart(draw_chart(df, series, ticker, company_name), use_container_width=True)

        col_rsi, col_macd = st.columns(2)
        with col_rsi:
            st.caption("RSI (14日)")
            st.plotly_chart(draw_rsi(series), use_container_width=True)
        with col_macd:
            st.caption("MACD (12-26-9)")
            st.plotly_chart(draw_macd(series), use_container_width=True)

        col_stoch, col_atr = st.columns(2)
        with col_stoch:
            st.caption("ストキャスティクス (14, 3)")
            st.plotly_chart(draw_stochastics(series), use_container_width=True)
        with col_atr:
            st.caption("ATR (14日) ― 値動きのボラティリティ")
            st.plotly_chart(draw_atr(series), use_container_width=True)

        st.caption("出来高")
        st.plotly_chart(draw_volume(df), use_container_width=True)

        st.info(
            f"📐 **ATRベースの損切り目安**　"
            f"買いポジション: **{ind['stop_loss_buy']:,.2f}** 以下で損切り　／　"
            f"売りポジション: **{ind['stop_loss_sell']:,.2f}** 以上で損切り"
        )

        st.markdown("---")
        st.subheader(f"🤖 テクニカル分析レポート：{ticker}（{company_name}）")
        client       = Anthropic(api_key=api_key)
        response_box = st.empty()
        full_text    = ""
        with st.spinner("Claude が分析中…"):
            for chunk in ask_claude_stream(client, ticker, company_name, ind):
                full_text += chunk
                response_box.markdown(full_text + "▌")
        response_box.markdown(full_text)

        add_to_history(ticker, company_name, ind, full_text)
        st.success("✅ 分析結果を履歴に保存しました")

    elif analyze_btn and not ticker_input:
        st.warning("銘柄コードを入力してください。")

# ══════════════════════════════════════════════════════════════════
# タブ②：スクリーニング
# ══════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("🎯 スクリーニング結果")

    # ── 自動スキャン結果（毎朝保存されたデータ）──────────────────
    scan_today = load_scan(SCAN_TODAY_FILE)

    if scan_today:
        st.success(f"📅 最終スキャン: {scan_today['scanned_at']}　対象: {scan_today['total']}銘柄")

        inner_tab1, inner_tab2 = st.tabs(["🔔 アラート（上位3銘柄）", "📊 前日比較（変化5銘柄）"])

        # ── アラートタブ ─────────────────────────────────────────
        with inner_tab1:
            st.caption(f"スコア{ALERT_SCORE}点以上の上位{ALERT_TOP_N}銘柄を表示します")
            if alert_stocks:
                for r in alert_stocks:
                    ind = r["ind"]
                    with st.container(border=True):
                        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 2])
                        with col_a:
                            st.markdown(f"### 🟢 {r['ticker']}")
                            st.caption(r["company_name"])
                        with col_b:
                            st.metric("スコア", f"{r['score']}/8")
                        with col_c:
                            chg_color = "normal" if ind["change_pct"] >= 0 else "inverse"
                            st.metric("現在値", f"{ind['price']:,.2f}",
                                      f"{ind['change_pct']:+.2f}%", delta_color=chg_color)
                        with col_d:
                            for sig in r["signals"]:
                                st.caption(f"✅ {sig}")
                        if st.button(f"🔍 {r['ticker']} を詳細分析",
                                     key=f"alert_detail_{r['ticker']}", use_container_width=True):
                            st.session_state["watchlist_trigger"] = r["ticker"]
                            st.rerun()
            else:
                st.info(f"本日はスコア{ALERT_SCORE}点以上の銘柄はありませんでした。")

        # ── 前日比較タブ ─────────────────────────────────────────
        with inner_tab2:
            changes, today_at, prev_at = get_score_changes()
            if changes:
                st.caption(f"前回スキャンから最も上昇した{COMPARE_TOP_N}銘柄　（前回: {prev_at} → 今回: {today_at}）")
                for r in changes:
                    ind = r["ind"]
                    with st.container(border=True):
                        col_a, col_b, col_c, col_d = st.columns([3, 2, 1, 2])
                        with col_a:
                            st.markdown(f"### ⬆️ {r['ticker']}")
                            st.caption(r["company_name"])
                        with col_b:
                            st.metric("スコア変化",
                                      f"{r['score']}/8",
                                      f"+{r['diff']} （前回 {r['prev_score']}点）")
                        with col_c:
                            chg_color = "normal" if ind["change_pct"] >= 0 else "inverse"
                            st.metric("現在値", f"{ind['price']:,.2f}",
                                      f"{ind['change_pct']:+.2f}%", delta_color=chg_color)
                        with col_d:
                            for sig in r["signals"]:
                                st.caption(f"✅ {sig}")
                        if st.button(f"🔍 {r['ticker']} を詳細分析",
                                     key=f"cmp_detail_{r['ticker']}", use_container_width=True):
                            st.session_state["watchlist_trigger"] = r["ticker"]
                            st.rerun()
            elif load_scan(SCAN_PREV_FILE):
                st.info("前回からスコアが上昇した銘柄はありませんでした。")
            else:
                st.info("前回のスキャンデータがまだありません。2回目のスキャン後に比較できます。")

    else:
        st.warning("自動スキャンのデータがまだありません。GitHub Actions が毎朝5時に自動実行します。")

    # ── 手動スキャン（ウォッチリストのみ） ───────────────────────
    st.markdown("---")
    st.subheader("🔄 今すぐスキャン（ウォッチリストのみ）")
    st.caption("GitHub Actionsを待たずに、今すぐウォッチリストだけをスキャンできます。")

    watchlist_scan = load_watchlist()
    if not watchlist_scan:
        st.warning("ウォッチリストに銘柄が登録されていません。サイドバーから追加してください。")
    else:
        with st.expander("📖 スコアの見方"):
            st.markdown("""
| スコア | 判定 |
|--------|------|
| 6〜8点 | 🟢 強い買いシグナル |
| 4〜5点 | 🟡 やや買い寄り |
| 2〜3点 | ⚪ 中立 |
| 0〜1点 | 🔴 シグナル弱 |
            """)

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            min_score = st.slider("表示する最低スコア", 0, 8, 3)
        with col_opt2:
            use_claude = st.checkbox("Claudeで精密分析", value=True,
                                     help="スコア上位をClaudeが解説（API使用）")

        if st.button("🚀 今すぐスキャン", type="primary", use_container_width=True, key="manual_scan_btn"):
            if use_claude and not api_key:
                st.error("Claudeでの分析にはAPIキーが必要です。")
                st.stop()

            results  = run_screening(watchlist_scan, period)
            filtered = [r for r in results if not r["error"] and r["score"] >= min_score]
            errors   = [r for r in results if r["error"]]

            st.markdown("---")
            st.subheader("📊 スキャン結果")

            if not filtered:
                st.info(f"スコア{min_score}点以上の銘柄が見つかりませんでした。")
            else:
                for r in filtered:
                    score = r["score"]
                    badge = "🟢" if score >= 6 else "🟡" if score >= 4 else "⚪"
                    ind   = r["ind"]
                    with st.container(border=True):
                        col_a, col_b, col_c, col_d, col_e = st.columns([3, 1, 1, 1, 2])
                        with col_a:
                            st.markdown(f"**{badge} {r['label']}　`{r['ticker']}`**")
                            st.caption(r["company_name"])
                        with col_b:
                            st.metric("スコア", f"{score}/8")
                        with col_c:
                            chg_color = "normal" if ind["change_pct"] >= 0 else "inverse"
                            st.metric("現在値", f"{ind['price']:,.2f}",
                                      f"{ind['change_pct']:+.2f}%", delta_color=chg_color)
                        with col_d:
                            st.metric("RSI", f"{ind['rsi']:.1f}")
                        with col_e:
                            for emoji, name, _ in r["signals"]:
                                st.caption(f"{emoji} {name}")
                        if st.button(f"🔍 {r['ticker']} を詳細分析する",
                                     key=f"manual_detail_{r['ticker']}", use_container_width=True):
                            st.session_state["watchlist_trigger"] = r["ticker"]
                            st.rerun()

            if errors:
                with st.expander(f"⚠️ データ取得失敗: {len(errors)}銘柄"):
                    for r in errors:
                        st.caption(f"・{r['ticker']}（{r['label']}）")

            top_candidates = [r for r in filtered if r["score"] >= 4]
            if use_claude and top_candidates:
                st.markdown("---")
                st.subheader("🤖 Claude による買い候補の総評")
                client   = Anthropic(api_key=api_key)
                ai_box   = st.empty()
                ai_text  = ""
                with st.spinner("Claude が分析中…"):
                    for chunk in ask_claude_screening(client, top_candidates):
                        ai_text += chunk
                        ai_box.markdown(ai_text + "▌")
                ai_box.markdown(ai_text)
