"""
fetch_data.py — Pull 60 days of 5-minute OHLCV from Binance Futures (public API).

Dependencies: requests, pandas  (stdlib: time, datetime)
Usage: python fetch_data.py
Output: training_data.csv
"""

import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS    = ["FLOWUSDT", "ADAUSDT", "XRPUSDT", "TIAUSDT", "LTCUSDT", "AVAXUSDT"]
INTERVAL   = "5m"
DAYS_BACK  = 120
LIMIT      = 1000          # max candles per request (Binance cap)
SLEEP_SEC  = 0.25          # pause between HTTP calls to stay well under rate limits
OUTPUT     = "training_data.csv"

MS_PER_CANDLE = 5 * 60 * 1000   # 5 minutes in milliseconds


def fetch_symbol(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """
    Page through [start_ms, end_ms) in LIMIT-sized windows and return every
    raw kline row Binance sends back.  Each row is the native list format:
      [open_time, open, high, low, close, volume, close_time, ...]
    """
    rows = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": cursor,
            "endTime":   end_ms - 1,   # inclusive upper bound
            "limit":     LIMIT,
        }
        resp = requests.get(BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])

        # Advance cursor past the last candle we received.
        cursor = last_open_time + MS_PER_CANDLE

        # Binance returned fewer candles than LIMIT → we have everything.
        if len(batch) < LIMIT:
            break

        time.sleep(SLEEP_SEC)

    return rows


def main() -> None:
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - DAYS_BACK * 24 * 60 * 60 * 1000

    print(
        f"Fetching {INTERVAL} candles for {len(SYMBOLS)} symbols "
        f"from {datetime.utcfromtimestamp(start_ms / 1000).strftime('%Y-%m-%d')} UTC "
        f"to now …"
    )

    all_frames: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        print(f"  [{symbol}] requesting …", end=" ", flush=True)
        raw = fetch_symbol(symbol, start_ms, now_ms)
        print(f"{len(raw)} candles")

        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        df["symbol"]    = symbol
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        all_frames.append(df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])

        time.sleep(SLEEP_SEC)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(["symbol", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(combined):,} rows → {OUTPUT}")
    print(combined.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
