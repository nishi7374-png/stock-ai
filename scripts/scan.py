"""
自動スキャンスクリプト
GitHub Actions から毎朝実行される。
日経225 + ウォッチリストをスキャンして結果を保存する。
"""

import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# ─── 日経225の全銘柄リスト ────────────────────────────────────────
NIKKEI225_TICKERS = [
    "1332.T","1333.T","1375.T","1376.T","1377.T","1379.T","1382.T","1383.T",
    "1384.T","1385.T","1386.T","1387.T","1388.T","1605.T","1662.T","1718.T",
    "1720.T","1721.T","1801.T","1802.T","1803.T","1808.T","1812.T","1925.T",
    "1928.T","1963.T","2002.T","2004.T","2006.T","2007.T","2008.T","2009.T",
    "2010.T","2011.T","2013.T","2014.T","2015.T","2016.T","2108.T","2109.T",
    "2120.T","2121.T","2122.T","2124.T","2127.T","2128.T","2130.T","2175.T",
    "2181.T","2183.T","2186.T","2193.T","2201.T","2202.T","2206.T","2207.T",
    "2208.T","2215.T","2217.T","2220.T","2221.T","2222.T","2224.T","2226.T",
    "2229.T","2264.T","2267.T","2268.T","2269.T","2270.T","2281.T","2282.T",
    "2284.T","2285.T","2288.T","2292.T","2294.T","2296.T","2301.T","2303.T",
    "2305.T","2307.T","2309.T","2315.T","2317.T","2321.T","2326.T","2327.T",
    "2329.T","2330.T","2331.T","2332.T","2334.T","2335.T","2336.T","2337.T",
    "2338.T","2340.T","2341.T","2344.T","2345.T","2349.T","2351.T","2353.T",
    "2354.T","2355.T","2359.T","2374.T","2375.T","2376.T","2377.T","2378.T",
    "2379.T","2384.T","2385.T","2388.T","2389.T","2391.T","2393.T","2395.T",
    "2397.T","2398.T","2406.T","2408.T","2411.T","2412.T","2413.T","2415.T",
    "2417.T","2418.T","2419.T","2420.T","2424.T","2425.T","2427.T","2428.T",
    "2429.T","2432.T","2433.T","2434.T","2436.T","2437.T","2438.T","2440.T",
    "2442.T","2443.T","2444.T","2445.T","2446.T","2447.T","2449.T","2450.T",
    "2453.T","2454.T","2459.T","2460.T","2461.T","2462.T","2463.T","2464.T",
    "2465.T","2471.T","2472.T","2477.T","2479.T","2480.T","2483.T","2484.T",
    "2485.T","2489.T","2492.T","2493.T","2497.T","2498.T","2499.T","2501.T",
    "2502.T","2503.T","2531.T","2533.T","2579.T","2587.T","2593.T","2594.T",
    "2597.T","2602.T","2607.T","2613.T","2614.T","2651.T","2653.T","2659.T",
    "2670.T","2674.T","2681.T","2685.T","2686.T","2695.T","2698.T","2702.T",
    "2712.T","2726.T","2729.T","2730.T","2733.T","2736.T","2737.T","2742.T",
    "2751.T","2752.T","2753.T","2760.T","2764.T","2767.T","2768.T","2778.T",
    "2784.T","2788.T",
]

# 正式な日経225構成銘柄（2024年時点）
NIKKEI225_OFFICIAL = [
    "1332.T","1605.T","1721.T","1801.T","1802.T","1803.T","1808.T","1812.T",
    "1925.T","1928.T","1963.T","2002.T","2269.T","2282.T","2413.T","2432.T",
    "2501.T","2502.T","2503.T","2531.T","2579.T","2768.T","2801.T","2802.T",
    "2871.T","2914.T","3086.T","3092.T","3099.T","3197.T","3289.T","3382.T",
    "3401.T","3402.T","3405.T","3407.T","3436.T","3659.T","3861.T","3863.T",
    "4004.T","4005.T","4021.T","4042.T","4043.T","4061.T","4063.T","4151.T",
    "4183.T","4188.T","4208.T","4324.T","4452.T","4502.T","4503.T","4506.T",
    "4507.T","4519.T","4523.T","4543.T","4568.T","4578.T","4631.T","4661.T",
    "4689.T","4704.T","4751.T","4755.T","4901.T","4902.T","5019.T","5020.T",
    "5101.T","5108.T","5201.T","5202.T","5214.T","5233.T","5301.T","5332.T",
    "5333.T","5401.T","5406.T","5411.T","5541.T","5631.T","5706.T","5707.T",
    "5711.T","5713.T","5714.T","5715.T","5801.T","5802.T","5803.T","5901.T",
    "6098.T","6103.T","6113.T","6178.T","6301.T","6302.T","6305.T","6326.T",
    "6361.T","6367.T","6406.T","6412.T","6471.T","6472.T","6473.T","6479.T",
    "6501.T","6503.T","6504.T","6506.T","6526.T","6532.T","6594.T","6645.T",
    "6674.T","6701.T","6702.T","6703.T","6706.T","6723.T","6724.T","6725.T",
    "6726.T","6727.T","6728.T","6752.T","6753.T","6755.T","6758.T","6762.T",
    "6770.T","6773.T","6774.T","6776.T","6778.T","6781.T","6789.T","6796.T",
    "6857.T","6861.T","6869.T","6902.T","6952.T","6954.T","6971.T","6976.T",
    "6981.T","6988.T","7003.T","7004.T","7011.T","7012.T","7013.T","7186.T",
    "7201.T","7202.T","7203.T","7205.T","7211.T","7261.T","7267.T","7269.T",
    "7270.T","7272.T","7731.T","7733.T","7735.T","7741.T","7751.T","7752.T",
    "7762.T","7832.T","7911.T","7912.T","7951.T","8001.T","8002.T","8003.T",
    "8005.T","8006.T","8007.T","8008.T","8010.T","8011.T","8012.T","8015.T",
    "8016.T","8031.T","8035.T","8053.T","8056.T","8058.T","8059.T","8060.T",
    "8061.T","8088.T","8101.T","8113.T","8233.T","8252.T","8253.T","8267.T",
    "8303.T","8304.T","8306.T","8308.T","8309.T","8316.T","8331.T","8354.T",
    "8355.T","8411.T","8601.T","8604.T","8628.T","8630.T","8697.T","8725.T",
    "8750.T","8766.T","8795.T","9001.T","9005.T","9007.T","9008.T","9009.T",
    "9020.T","9021.T","9022.T","9064.T","9101.T","9104.T","9107.T","9202.T",
    "9301.T","9432.T","9433.T","9434.T","9501.T","9502.T","9503.T","9531.T",
    "9532.T","9602.T","9613.T","9735.T","9766.T","9983.T","9984.T",
]

# ─── パス設定 ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WATCHLIST_FILE  = ROOT / "watchlist.json"
CONFIG_FILE     = ROOT / "config.json"
SCAN_TODAY_FILE = ROOT / "scan_today.json"
SCAN_PREV_FILE  = ROOT / "scan_prev.json"

# ─── ウォッチリスト読み込み ───────────────────────────────────────
def load_watchlist_tickers():
    if WATCHLIST_FILE.exists():
        try:
            wl = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            return [w["ticker"] for w in wl]
        except Exception:
            pass
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return cfg.get("tickers", [])
        except Exception:
            pass
    return []

# ─── テクニカル指標 ───────────────────────────────────────────────
def fetch_and_score(ticker: str):
    """1銘柄を取得してスコアを返す。失敗したらNoneを返す。"""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="6mo", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
       　　 df.index = df.index.tz_localize(None)
   　　 df = df.dropna(subset=["Close"])
   　　 if df is None or df.empty or len(df) < 26:
       　　 return None

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # 移動平均
        ma5  = close.rolling(5).mean()
        ma25 = close.rolling(25).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig

        # ボリンジャー
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # ストキャス
        ll = low.rolling(14).min()
        hh = high.rolling(14).max()
        stk = (close - ll) / (hh - ll).replace(0, np.nan) * 100
        std = stk.rolling(3).mean()

        ind = {
            "price":      float(close.iloc[-1]),
            "change_pct": float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100),
            "ma5":        float(ma5.iloc[-1]),
            "ma25":       float(ma25.iloc[-1]),
            "rsi":        float(rsi.iloc[-1]),
            "macd":       float(macd.iloc[-1]),
            "macd_sig":   float(sig.iloc[-1]),
            "macd_hist":  float(hist.iloc[-1]),
            "macd_prev":  float(hist.iloc[-2]),
            "volume":     float(volume.iloc[-1]),
            "volume_ma5": float(volume.rolling(5).mean().iloc[-1]),
            "bb_upper":   float(bb_upper.iloc[-1]),
            "bb_lower":   float(bb_lower.iloc[-1]),
            "stoch_k":    float(stk.iloc[-1]),
            "stoch_d":    float(std.iloc[-1]),
        }

        # スコアリング
        score = 0
        signals = []

        if 30 <= ind["rsi"] <= 50:
            score += 1; signals.append("RSI回復中")
        elif ind["rsi"] < 30:
            score += 0.5; signals.append("RSI売られすぎ")
        if ind["macd_hist"] > ind["macd_prev"]:
            score += 1; signals.append("MACD改善")
        if ind["macd"] > ind["macd_sig"]:
            score += 1; signals.append("MACDゴールデンクロス")
        if ind["price"] > ind["ma25"]:
            score += 1; signals.append("MA25上抜け")
        if ind["ma5"] > ind["ma25"]:
            score += 1; signals.append("短期MA上昇")
        if ind["stoch_k"] > ind["stoch_d"] and 20 <= ind["stoch_k"] <= 80:
            score += 1; signals.append("ストキャス上昇")
        if ind["volume"] > ind["volume_ma5"]:
            score += 1; signals.append("出来高増加")
        bb_range = ind["bb_upper"] - ind["bb_lower"]
        if bb_range > 0:
            bb_pos = (ind["price"] - ind["bb_lower"]) / bb_range
            if bb_pos < 0.3:
                score += 1; signals.append("BB下限付近")

        # 会社名取得（失敗しても続行）
        name = ticker
        try:
            info = tk.info
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            pass

        return {
            "ticker":       ticker,
            "company_name": name,
            "score":        score,
            "signals":      signals,
            "ind":          ind,
        }
    except Exception:
        return None


def run_scan():
    # スキャン対象：日経225 ＋ ウォッチリスト（重複排除）
    watchlist_tickers = load_watchlist_tickers()
    all_tickers = list(dict.fromkeys(NIKKEI225_OFFICIAL + watchlist_tickers))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] スキャン開始: {len(all_tickers)}銘柄")

    results = []
    for i, ticker in enumerate(all_tickers):
        r = fetch_and_score(ticker)
        if r:
            results.append(r)
        # Yahoo Finance への負荷軽減のため少し待つ
        time.sleep(0.3)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_tickers)} 完了...")

    # スコア降順でソート
    results.sort(key=lambda x: x["score"], reverse=True)

    # 前回の結果を退避
    if SCAN_TODAY_FILE.exists():
        SCAN_TODAY_FILE.rename(SCAN_PREV_FILE)

    # 今回の結果を保存
    SCAN_TODAY_FILE.write_text(
        json.dumps({
            "scanned_at": datetime.now().strftime("%Y/%m/%d %H:%M"),
            "total":      len(results),
            "results":    results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 完了: {len(results)}銘柄保存")


if __name__ == "__main__":
    run_scan()
