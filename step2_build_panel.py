r"""
Milestone 1 - Step 2: build one clean 1-minute table for all 5 coins.

Reads level1 (quotes) + trade files for BTC, ETH, SOL, XRP, DOGE,
keeps the columns we need, and saves one file per coin to
    <this folder>\data\<COIN>_1min.pkl
Every later script loads these files instead of the raw .csv.gz files.

Run from PowerShell:
    python "D:\BLOCKCHAIN PROJECT\step2_build_panel.py"

Takes several minutes (about 10,000 small files). Progress is printed.
"""
import os
import time

import numpy as np
import pandas as pd

ROOT = r"D:\Fordham\data\fordham\binance\futures\latest"   # <-- change if your raw course data lives elsewhere
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # <repo>/data
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

L1_COLS = ["ts_end", "carryover", "close_mid", "close_bid", "close_ask",
           "close_bid_size", "close_ask_size", "median_spread", "tick_volatility"]
TR_COLS = ["ts_end", "trade_count", "volume", "buy_volume", "sell_volume", "vwap"]

os.makedirs(OUT, exist_ok=True)


def load_folder(kind, suffix, coin, cols):
    folder = os.path.join(ROOT, kind)
    days = sorted(d for d in os.listdir(folder) if d.isdigit())
    frames, missing = [], 0
    for i, day in enumerate(days):
        f = os.path.join(folder, day, f"{coin}.{day}.{suffix}.1min.csv.gz")
        if not os.path.exists(f):
            missing += 1
            continue
        try:
            frames.append(pd.read_csv(f, usecols=cols))
        except Exception as e:
            print(f"   ! could not read {f}: {e}")
            missing += 1
        if (i + 1) % 200 == 0:
            print(f"   {kind}: {i + 1}/{len(days)} days read")
    df = pd.concat(frames, ignore_index=True)
    # ts_end is milliseconds; round to the minute so quotes and trades line up
    df["time"] = pd.to_datetime(df["ts_end"].astype("int64"), unit="ms", utc=True).dt.floor("min")
    df = df.drop(columns="ts_end").drop_duplicates("time", keep="last").set_index("time")
    return df, missing


for coin in COINS:
    t0 = time.time()
    print(f"\n=== {coin} ===")
    l1, miss_l1 = load_folder("level1_1min", "level1", coin, L1_COLS)
    tr, miss_tr = load_folder("trade_1min", "trade", coin, TR_COLS)

    df = l1.join(tr, how="outer")
    df["dollar_volume"] = df["volume"] * df["vwap"]
    df["spread_bps"] = 1e4 * (df["close_ask"] - df["close_bid"]) / df["close_mid"]

    # Put on a complete minute grid so gaps are visible (NaN) instead of hidden
    full = pd.date_range(df.index.min(), df.index.max(), freq="min", tz="UTC")
    df = df.reindex(full)
    df.index.name = "time"

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    path = os.path.join(OUT, f"{coin}_1min.pkl")
    df.to_pickle(path)

    print(f"Saved {path}")
    print(f"  Minutes: {len(df):,}   from {df.index.min()} to {df.index.max()}")
    print(f"  Missing days  level1: {miss_l1}   trade: {miss_tr}")
    print(f"  Minutes with no mid price: {df['close_mid'].isna().mean():.2%}")
    print(f"  Minutes flagged carryover (stale quote): {(df['carryover'] == 1).mean():.2%}")
    print(f"  Median spread: {df['spread_bps'].median():.2f} bps")
    print(f"  Mean daily $ volume: {df['dollar_volume'].resample('D').sum().mean() / 1e9:.2f} B")
    print(f"  Took {time.time() - t0:.0f} s")

print("\nDone. Paste this whole output back.")
