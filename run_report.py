#!/usr/bin/env python3
"""Run all scenarios and generate HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_panel_with_ohlc
from src.multi_exchange_data import align_panel, load_multi_exchange_panel
from src.report import run_all_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache = root / "data" / "cache"
    results = root / "results"

    print("=== Single-exchange data (Binance) ===")
    single = load_panel_with_ohlc(days=args.days, cache_dir=cache)

    print("=== Multi-exchange data (Binance, Bybit, OKX) ===")
    cross = load_multi_exchange_panel(days=args.days, cache_dir=cache / "multi")
    cross = align_panel(cross)
    print(
        f"  aligned bars: {cross['open_time'].nunique()}, "
        f"exchanges: {list(cross['exchange'].unique())}"
    )

    summaries = run_all_scenarios(single, cross, results)

    print("\n=== DONE ===")
    for s in summaries:
        print(
            f"  {s['scenario']}: {s.get('total_return_pct', 0):.2f}% "
            f"({s.get('trades_executed', 0)} trades)"
        )
    print(f"\nOpen report: {results / 'REPORT.html'}")


if __name__ == "__main__":
    main()
