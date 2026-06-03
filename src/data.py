"""Download Binance spot 5m klines."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# Liquid USDT pairs for multi-currency cycles (presentation: BTC, ETH, SOL + others)
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
]


def symbol_to_base_quote(symbol: str) -> tuple[str, str]:
    if symbol.endswith("USDT"):
        return symbol[:-4], "USDT"
    raise ValueError(f"Unsupported symbol format: {symbol}")


def fetch_klines(
    symbol: str,
    interval: str = "5m",
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    params: Dict = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    r = requests.get(BINANCE_KLINES, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["symbol"] = symbol
    return df


def fetch_klines_range(
    symbol: str,
    interval: str = "5m",
    start_ms: int = 0,
    end_ms: Optional[int] = None,
    pause: float = 0.15,
) -> pd.DataFrame:
    """Paginate Binance klines (max 1000 per request)."""
    if end_ms is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    parts: List[pd.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk = fetch_klines(symbol, interval, start_ms=cursor, end_ms=end_ms, limit=1000)
        if chunk.empty:
            break
        parts.append(chunk)
        last_close = int(chunk["close_time"].iloc[-1])
        next_cursor = last_close + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(pause)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return out


def load_or_download_panel(
    symbols: Optional[List[str]] = None,
    days: int = 14,
    interval: str = "5m",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return long DataFrame: open_time, symbol, base, quote, close.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    end = datetime.now(timezone.utc)
    start_ms = int((end.timestamp() - days * 86400) * 1000)
    end_ms = int(end.timestamp() * 1000)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sym in symbols:
        cache_path = cache_dir / f"{sym}_{interval}_{days}d.csv" if cache_dir else None
        if cache_path is not None and cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=["open_time"])
        else:
            df = fetch_klines_range(sym, interval, start_ms, end_ms)
            if cache_path is not None and not df.empty:
                df.to_csv(cache_path, index=False)
        if not df.empty:
            frames.append(df[["open_time", "symbol", "close"]])

    if not frames:
        raise RuntimeError("No market data downloaded")

    panel = pd.concat(frames, ignore_index=True)
    panel["base"], panel["quote"] = zip(*panel["symbol"].map(symbol_to_base_quote))
    return panel


def load_panel_with_ohlc(
    symbols: Optional[List[str]] = None,
    days: int = 14,
    interval: str = "5m",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Like load_or_download_panel but keeps high/low for bid-ask proxy."""
    symbols = symbols or DEFAULT_SYMBOLS
    end = datetime.now(timezone.utc)
    start_ms = int((end.timestamp() - days * 86400) * 1000)
    end_ms = int(end.timestamp() * 1000)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sym in symbols:
        cache_path = cache_dir / f"{sym}_{interval}_{days}d.csv" if cache_dir else None
        if cache_path is not None and cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=["open_time"])
        else:
            df = fetch_klines_range(sym, interval, start_ms, end_ms)
            if cache_path is not None and not df.empty:
                df.to_csv(cache_path, index=False)
        if not df.empty:
            frames.append(df[["open_time", "symbol", "open", "high", "low", "close"]])

    if not frames:
        raise RuntimeError("No market data downloaded")
    panel = pd.concat(frames, ignore_index=True)
    panel["base"], panel["quote"] = zip(*panel["symbol"].map(symbol_to_base_quote))
    return panel


def panel_close_matrix(panel: pd.DataFrame, t: pd.Timestamp) -> Dict[str, float]:
    """symbol -> close at timestamp t."""
    snap = panel.loc[panel["open_time"] == t]
    return dict(zip(snap["symbol"], snap["close"]))
