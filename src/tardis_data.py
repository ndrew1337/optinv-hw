"""Free Tardis.dev L2 order-book samples (first day of each month, no API key).

Downloads `book_snapshot_25` CSV.gz from datasets.tardis.dev, stream-parses it
keeping only the top-N levels, and downsamples to a fixed time grid so the
event-driven backtest stays tractable.

Free tier: the FIRST DAY OF EACH MONTH is downloadable without an API key, for
all data types incl. full L2 order book. See https://docs.tardis.dev/faq/data
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DATASETS_BASE = "https://datasets.tardis.dev/v1"

# Tardis exchange ids differ from ours: OKX is "okex", Bybit spot is "bybit-spot".
EXCHANGE_ID = {
    "binance": "binance",
    "bybit": "bybit-spot",
    "okx": "okex",
}


def tardis_symbol(exchange: str, asset: str) -> str:
    """okex uses BASE-QUOTE (BTC-USDT); binance/bybit use BASEQUOTE (BTCUSDT)."""
    return f"{asset}-USDT" if exchange == "okx" else f"{asset}USDT"


def dataset_url(exchange: str, asset: str, date: str, data_type: str = "book_snapshot_25") -> str:
    y, m, d = date.split("-")
    ex_id = EXCHANGE_ID[exchange]
    sym = tardis_symbol(exchange, asset)
    return f"{DATASETS_BASE}/{ex_id}/{data_type}/{y}/{m}/{d}/{sym}.csv.gz"


def _level_cols(depth: int) -> List[str]:
    cols = ["timestamp"]
    for i in range(depth):
        cols += [f"asks[{i}].price", f"asks[{i}].amount",
                 f"bids[{i}].price", f"bids[{i}].amount"]
    return cols


def download_raw(
    exchange: str,
    asset: str,
    date: str,
    raw_dir: Path,
    data_type: str = "book_snapshot_25",
) -> Optional[Path]:
    """Download the gzip CSV if not already cached. Returns path or None on failure."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    sym = tardis_symbol(exchange, asset)
    out = raw_dir / f"{EXCHANGE_ID[exchange]}_{sym}_{data_type}_{date.replace('-', '')}.csv.gz"
    if out.exists() and out.stat().st_size > 1024:
        return out
    url = dataset_url(exchange, asset, date, data_type)
    try:
        urllib.request.urlretrieve(url, out)  # noqa: S310 (trusted host)
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed {exchange}/{asset}: {exc}")
        return None
    if out.stat().st_size < 1024:
        print(f"  too small, likely no data: {exchange}/{asset} ({out.stat().st_size}b)")
        return None
    return out


def load_grid(
    exchange: str,
    asset: str,
    date: str,
    raw_dir: Path,
    grid_dir: Path,
    depth: int = 5,
    grid_ms: int = 1000,
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame on a fixed `grid_ms` time grid with top-`depth` levels.

    A `grid_ms` of e.g. 100 lets the backtest model latency at the millisecond
    scale that arbitrage actually lives at (1s is far too coarse). Stream-parses
    the raw gz keeping the LAST snapshot within each grid bucket. The time column
    `ts` is in MILLISECONDS. Cached as compact parquet.
    """
    grid_dir.mkdir(parents=True, exist_ok=True)
    cache = grid_dir / f"{exchange}_{asset}_{date.replace('-', '')}_d{depth}_g{grid_ms}ms.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        # Older cache files used seconds in a `sec` column. Normalize them so
        # callers can safely request the same cache without schema surprises.
        if "ts" not in df.columns and "sec" in df.columns:
            df = df.rename(columns={"sec": "ts"})
            df["ts"] = (df["ts"].astype("int64") * 1000).astype("int64")
        if "ts" not in df.columns:
            raise ValueError(f"Tardis cache is missing a ts column: {cache}")
        return df

    raw = download_raw(exchange, asset, date, raw_dir)
    if raw is None:
        return None

    usecols = _level_cols(depth)
    bucket_us = grid_ms * 1_000  # microseconds per grid bucket
    parts: List[pd.DataFrame] = []

    reader = pd.read_csv(raw, usecols=usecols, chunksize=1_000_000, compression="gzip")
    for chunk in reader:
        chunk["bucket"] = chunk["timestamp"] // bucket_us
        # keep last row per bucket within this chunk (vectorised, no iterrows)
        parts.append(chunk.groupby("bucket", sort=False).tail(1))

    if not parts:
        return None

    # concat preserves chunk order -> final tail(1) per bucket is the global last
    df = pd.concat(parts, ignore_index=True)
    df = df.groupby("bucket", sort=True).tail(1).reset_index(drop=True)
    df["ts"] = (df["bucket"] * grid_ms).astype("int64")  # milliseconds
    df["exchange"] = exchange
    df["asset"] = asset
    df = df.drop(columns=["bucket", "timestamp"]).sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df


def load_l2_panels(
    assets: List[str],
    exchanges: List[str],
    date: str,
    raw_dir: Path,
    grid_dir: Path,
    depth: int = 5,
    grid_ms: int = 1000,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load {(exchange, asset): grid DataFrame} for every available combination."""
    panels: Dict[Tuple[str, str], pd.DataFrame] = {}
    for ex in exchanges:
        for a in assets:
            g = load_grid(ex, a, date, raw_dir, grid_dir, depth, grid_ms)
            if g is not None and not g.empty:
                panels[(ex, a)] = g
                print(f"  loaded {ex}/{a}: {len(g)} grid points")
            else:
                print(f"  no data {ex}/{a}")
    return panels
