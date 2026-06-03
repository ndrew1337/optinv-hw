"""Multi-exchange 5m OHLC download (Binance, Bybit, OKX)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .data import fetch_klines_range, symbol_to_base_quote

EXCHANGES = ("binance", "bybit", "okx")

# Pairs available on all three spot markets
CROSS_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "AVAXUSDT",
]


def _okx_inst_id(symbol: str) -> str:
    base, quote = symbol_to_base_quote(symbol)
    return f"{base}-{quote}"


def fetch_bybit_klines_range(
    symbol: str,
    interval: str = "5m",
    start_ms: int = 0,
    end_ms: Optional[int] = None,
    pause: float = 0.12,
) -> pd.DataFrame:
    if end_ms is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Bybit: 1=1m, 5=5m
    bybit_iv = "5" if interval == "5m" else "1"
    parts: List[pd.DataFrame] = []
    cursor = end_ms
    while cursor > start_ms:
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": bybit_iv,
            "end": cursor,
            "limit": 1000,
        }
        r = requests.get(
            "https://api.bybit.com/v5/market/kline", params=params, timeout=30
        )
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            break
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        parts.append(df)
        first_ms = int(rows[-1][0])
        if first_ms <= start_ms:
            break
        cursor = first_ms - 1
        time.sleep(pause)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    t0 = pd.to_datetime(start_ms, unit="ms", utc=True)
    out = out[out["open_time"] >= t0]
    out["symbol"] = symbol
    out["exchange"] = "bybit"
    return out


def fetch_okx_klines_range(
    symbol: str,
    interval: str = "5m",
    start_ms: int = 0,
    end_ms: Optional[int] = None,
    pause: float = 0.12,
) -> pd.DataFrame:
    if end_ms is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    inst = _okx_inst_id(symbol)
    parts: List[pd.DataFrame] = []
    cursor: Optional[str] = None
    while True:
        params: Dict = {"instId": inst, "bar": interval, "limit": "300"}
        if cursor:
            params["after"] = cursor
        r = requests.get(
            "https://www.okx.com/api/v5/market/candles", params=params, timeout=30
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            break
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vol_ccy",
                "vol_quote",
                "confirm",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        parts.append(df)
        oldest_ms = int(rows[-1][0])
        if oldest_ms <= start_ms:
            break
        cursor = rows[-1][0]
        time.sleep(pause)
        if len(rows) < 300:
            break

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    out = out[out["open_time"].astype("int64") // 10**6 >= start_ms]
    out["symbol"] = symbol
    out["exchange"] = "okx"
    return out


def fetch_exchange_klines(
    exchange: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    if exchange == "binance":
        df = fetch_klines_range(symbol, interval, start_ms, end_ms)
        if not df.empty:
            df["exchange"] = "binance"
        return df
    if exchange == "bybit":
        return fetch_bybit_klines_range(symbol, interval, start_ms, end_ms)
    if exchange == "okx":
        return fetch_okx_klines_range(symbol, interval, start_ms, end_ms)
    raise ValueError(f"Unknown exchange: {exchange}")


def load_multi_exchange_panel(
    symbols: Optional[List[str]] = None,
    exchanges: Optional[List[str]] = None,
    days: int = 14,
    interval: str = "5m",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    symbols = symbols or CROSS_SYMBOLS
    exchanges = list(exchanges or EXCHANGES)
    end = datetime.now(timezone.utc)
    start_ms = int((end.timestamp() - days * 86400) * 1000)
    end_ms = int(end.timestamp() * 1000)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for ex in exchanges:
        for sym in symbols:
            cache_path = (
                cache_dir / f"{ex}_{sym}_{interval}_{days}d.csv" if cache_dir else None
            )
            try:
                if cache_path is not None and cache_path.exists():
                    df = pd.read_csv(cache_path, parse_dates=["open_time"])
                else:
                    df = fetch_exchange_klines(ex, sym, interval, start_ms, end_ms)
                    if cache_path is not None and not df.empty:
                        df.to_csv(cache_path, index=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {ex} {sym}: {exc}")
                continue
            if df.empty:
                continue
            cols = ["open_time", "exchange", "symbol", "open", "high", "low", "close"]
            frames.append(df[cols])

    if not frames:
        raise RuntimeError("No multi-exchange data loaded")

    panel = pd.concat(frames, ignore_index=True)
    panel["base"], panel["quote"] = zip(*panel["symbol"].map(symbol_to_base_quote))
    return panel


def align_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep timestamps present on all exchanges (inner join on time)."""
    counts = panel.groupby("open_time")["exchange"].nunique()
    n_ex = panel["exchange"].nunique()
    valid_times = counts[counts >= n_ex].index
    return panel[panel["open_time"].isin(valid_times)].copy()
