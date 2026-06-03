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


def pair_key(base: str, quote: str) -> str:
    return f"{base.upper()}{quote.upper()}"


def normalize_pair(pair: str, default_quote: str = "USDT") -> Tuple[str, str]:
    pair = pair.strip().upper().replace("-", "/")
    if "/" in pair:
        base, quote = pair.split("/", 1)
        return base.strip(), quote.strip()
    if pair.endswith("USDT"):
        return pair[:-4], "USDT"
    return pair, default_quote.upper()


def tardis_pair_symbol(exchange: str, base: str, quote: str) -> str:
    """okex uses BASE-QUOTE (BTC-USDT); binance/bybit use BASEQUOTE (BTCUSDT)."""
    base = base.upper()
    quote = quote.upper()
    return f"{base}-{quote}" if exchange == "okx" else f"{base}{quote}"


def tardis_symbol(exchange: str, asset: str) -> str:
    """Backward-compatible ASSET/USDT symbol helper."""
    base, quote = normalize_pair(asset)
    return tardis_pair_symbol(exchange, base, quote)


def dataset_pair_url(
    exchange: str,
    base: str,
    quote: str,
    date: str,
    data_type: str = "book_snapshot_25",
) -> str:
    y, m, d = date.split("-")
    ex_id = EXCHANGE_ID[exchange]
    sym = tardis_pair_symbol(exchange, base, quote)
    return f"{DATASETS_BASE}/{ex_id}/{data_type}/{y}/{m}/{d}/{sym}.csv.gz"


def dataset_url(exchange: str, asset: str, date: str, data_type: str = "book_snapshot_25") -> str:
    base, quote = normalize_pair(asset)
    return dataset_pair_url(exchange, base, quote, date, data_type)


def _level_cols(depth: int) -> List[str]:
    cols = ["timestamp"]
    for i in range(depth):
        cols += [f"asks[{i}].price", f"asks[{i}].amount",
                 f"bids[{i}].price", f"bids[{i}].amount"]
    return cols


def download_pair_raw(
    exchange: str,
    base: str,
    quote: str,
    date: str,
    raw_dir: Path,
    data_type: str = "book_snapshot_25",
) -> Optional[Path]:
    """Download the gzip CSV if not already cached. Returns path or None on failure."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    sym = tardis_pair_symbol(exchange, base, quote)
    out = raw_dir / f"{EXCHANGE_ID[exchange]}_{sym}_{data_type}_{date.replace('-', '')}.csv.gz"
    if out.exists() and out.stat().st_size > 1024:
        return out
    url = dataset_pair_url(exchange, base, quote, date, data_type)
    try:
        urllib.request.urlretrieve(url, out)  # noqa: S310 (trusted host)
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed {exchange}/{base}{quote}: {exc}")
        return None
    if out.stat().st_size < 1024:
        print(f"  too small, likely no data: {exchange}/{base}{quote} ({out.stat().st_size}b)")
        return None
    return out


def download_raw(
    exchange: str,
    asset: str,
    date: str,
    raw_dir: Path,
    data_type: str = "book_snapshot_25",
) -> Optional[Path]:
    base, quote = normalize_pair(asset)
    return download_pair_raw(exchange, base, quote, date, raw_dir, data_type)


def load_pair_grid(
    exchange: str,
    base: str,
    quote: str,
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
    key = pair_key(base, quote)
    cache = grid_dir / f"{exchange}_{key}_{date.replace('-', '')}_d{depth}_g{grid_ms}ms.parquet"
    legacy_cache = grid_dir / f"{exchange}_{base}_{date.replace('-', '')}_d{depth}_g{grid_ms}ms.parquet"
    if not cache.exists() and quote == "USDT" and legacy_cache.exists():
        cache = legacy_cache
    if cache.exists():
        df = pd.read_parquet(cache)
        # Older cache files used seconds in a `sec` column. Normalize them so
        # callers can safely request the same cache without schema surprises.
        if "ts" not in df.columns and "sec" in df.columns:
            df = df.rename(columns={"sec": "ts"})
            df["ts"] = (df["ts"].astype("int64") * 1000).astype("int64")
        if "ts" not in df.columns:
            raise ValueError(f"Tardis cache is missing a ts column: {cache}")
        df["exchange"] = exchange
        df["asset"] = base if quote == "USDT" else key
        df["pair"] = key
        df["base"] = base
        df["quote"] = quote
        return df

    raw = download_pair_raw(exchange, base, quote, date, raw_dir)
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
    df["asset"] = base if quote == "USDT" else key
    df["pair"] = key
    df["base"] = base
    df["quote"] = quote
    df = df.drop(columns=["bucket", "timestamp"]).sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df


def load_grid(
    exchange: str,
    asset: str,
    date: str,
    raw_dir: Path,
    grid_dir: Path,
    depth: int = 5,
    grid_ms: int = 1000,
) -> Optional[pd.DataFrame]:
    base, quote = normalize_pair(asset)
    return load_pair_grid(exchange, base, quote, date, raw_dir, grid_dir, depth, grid_ms)


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


def load_l2_pair_panels(
    pairs: List[Tuple[str, str]],
    exchanges: List[str],
    date: str,
    raw_dir: Path,
    grid_dir: Path,
    depth: int = 5,
    grid_ms: int = 1000,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load {(exchange, pair_key): grid DataFrame} for arbitrary spot pairs."""
    panels: Dict[Tuple[str, str], pd.DataFrame] = {}
    for ex in exchanges:
        for base, quote in pairs:
            key = pair_key(base, quote)
            g = load_pair_grid(ex, base, quote, date, raw_dir, grid_dir, depth, grid_ms)
            if g is not None and not g.empty:
                panels[(ex, key)] = g
                print(f"  loaded {ex}/{key}: {len(g)} grid points")
            else:
                print(f"  no data {ex}/{key}")
    return panels
