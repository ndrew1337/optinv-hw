#!/usr/bin/env python3
"""Market diagnostics: theoretical cycles, fee sensitivity, spread stress-test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (
    BacktestConfig,
    build_graph_from_prices,
    mid_to_bid_ask,
    panel_close_matrix_local,
    run_backtest,
)
from src.bellman_ford import enumerate_triangles, find_best_cycle
from src.data import load_or_download_panel
from src.graph import ExchangeGraph


def scan_best_multiplier(
    panel: pd.DataFrame, fee: float, half_spread_bps: float = 5.0
) -> pd.DataFrame:
    rows = []
    for t in sorted(panel["open_time"].unique()):
        prices = panel_close_matrix_local(panel, t)
        ask, bid = mid_to_bid_ask(prices, half_spread_bps)
        g = build_graph_from_prices(prices, fee=fee, ask_prices=ask, bid_prices=bid)
        tris = enumerate_triangles(g)
        best_bf = find_best_cycle(g)
        best_mult = 1.0
        best_cycle = None
        if tris:
            best_mult = tris[0].gross_multiplier
            best_cycle = tris[0].currencies
        elif best_bf:
            best_mult = best_bf.gross_multiplier
            best_cycle = best_bf.currencies
        rows.append(
            {
                "time": t,
                "best_gross_mult": best_mult,
                "best_weight": np.log(best_mult) if best_mult > 0 else 0,
                "cycle": " -> ".join(best_cycle) if best_cycle else "",
            }
        )
    return pd.DataFrame(rows)


def spread_stress_test(base_price: float, spread_bps: float = 10.0) -> float:
    """
    Simulate triangular USDT-BTC-ETH with bid/ask on each leg.
    Returns best gross multiplier for USDT->BTC->ETH->USDT.
    """
    s = spread_bps / 10_000
    # Consistent mid prices
    btc = base_price
    eth_usdt = 3000.0
    eth_btc = eth_usdt / btc

    # Use the same graph convention as the backtest: price = quote per base.
    g = ExchangeGraph(fee=0.0)
    g.add_pair("BTC", "USDT", btc * (1 + s) / (1 - s))  # worsen rates via spread
    g.add_pair("ETH", "USDT", eth_usdt * (1 + s) / (1 - s))
    g.add_pair("ETH", "BTC", eth_btc * (1 + s) / (1 - s))
    tris = enumerate_triangles(g)
    return tris[0].gross_multiplier if tris else 1.0


def main() -> None:
    root = Path(__file__).resolve().parent
    cache = root / "data" / "cache"
    results = root / "results"
    results.mkdir(exist_ok=True)

    panel = load_or_download_panel(days=14, interval="5m", cache_dir=cache)

    print("Scanning triangles (mid prices, fee=0)...")
    diag0 = scan_best_multiplier(panel, fee=0.0, half_spread_bps=0.0)
    print(f"  max gross mult: {diag0['best_gross_mult'].max():.12f}")
    print(f"  bars with mult > 1+1e-9: {(diag0['best_gross_mult'] > 1.0 + 1e-9).sum()}")

    print("\nRealistic spread 5bps/side, fee=0.1%/leg...")
    diag1 = scan_best_multiplier(panel, fee=0.001, half_spread_bps=5.0)
    print(f"  max gross mult: {diag1['best_gross_mult'].max():.8f}")
    print(f"  bars with mult > 1: {(diag1['best_gross_mult'] > 1.0).sum()}")

    # Fee sweep on consistent synthetic edge
    print("\nFee sensitivity (synthetic 0.05% mispricing, 3-leg cycle):")
    g = ExchangeGraph(fee=0.0)
    btc = 50_000.0
    eth_usdt = 3_000.0
    mis = 1.0005
    g.add_pair("BTC", "USDT", btc)
    g.add_pair("ETH", "USDT", eth_usdt)
    g.add_pair("ETH", "BTC", eth_usdt / btc * mis)
    fee_sweep = {}
    for fee in [0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        g2 = ExchangeGraph(fee=fee)
        g2.add_pair("BTC", "USDT", btc)
        g2.add_pair("ETH", "USDT", eth_usdt)
        g2.add_pair("ETH", "BTC", eth_usdt / btc * mis)
        c = enumerate_triangles(g2)
        fee_sweep[fee] = c[0].gross_multiplier if c else 1.0
        print(f"  fee={fee*100:.2f}%  gross_mult={fee_sweep[fee]:.6f}")

    report = {
        "real_data_14d_5m": {
            "bars": int(panel["open_time"].nunique()),
            "pairs": int(panel["symbol"].nunique()),
            "max_gross_mult_fee0": float(diag0["best_gross_mult"].max()),
            "bars_profitable_fee0": int((diag0["best_gross_mult"] > 1.0).sum()),
            "max_gross_mult_fee01pct": float(diag1["best_gross_mult"].max()),
            "bars_profitable_fee01pct": int((diag1["best_gross_mult"] > 1.0).sum()),
        },
        "synthetic_mispricing_fee_sweep": fee_sweep,
        "spread_stress_bps10": float(spread_stress_test(50_000.0, 10.0)),
    }

    diag0.to_csv(results / "diagnostics_fee0.csv", index=False)
    with open(results / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Backtest with zero min threshold to see "paper" opportunities
    print("\nBacktest fee=0, mid only (no spread — theoretical)...")
    bt = run_backtest(
        panel,
        BacktestConfig(
            fee=0.0,
            slippage_bps=0.0,
            half_spread_bps=0.0,
            min_gross_multiplier=1.0000001,
        ),
    )
    print(bt.summary())

    print(f"\nFull report: {results}/analysis_report.json")


if __name__ == "__main__":
    main()
