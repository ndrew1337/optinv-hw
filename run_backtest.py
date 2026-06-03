#!/usr/bin/env python3
"""Download 5m Binance data, detect arbitrage cycles, run backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import BacktestConfig, run_backtest
from src.data import DEFAULT_SYMBOLS, load_or_download_panel, load_panel_with_ohlc


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Bellman-Ford arbitrage backtest")
    parser.add_argument("--days", type=int, default=14, help="History length in days")
    parser.add_argument("--fee", type=float, default=0.001, help="Fee per leg (0.001 = 0.1%)")
    parser.add_argument("--half-spread-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--all-cycles", action="store_true", help="Bellman-Ford any cycle length")
    parser.add_argument(
        "--intrabar-hilo",
        action="store_true",
        help="Optimistic: high/low as bid/ask (not realistic)",
    )
    parser.add_argument("--no-download", action="store_true", help="Use cache only")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache = root / "data" / "cache"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    print("Loading 5m Binance spot data...")
    loader = load_panel_with_ohlc if args.intrabar_hilo else load_or_download_panel
    panel = loader(
        symbols=DEFAULT_SYMBOLS,
        days=args.days,
        interval="5m",
        cache_dir=cache,
    )
    print(f"  bars: {panel['open_time'].nunique()}, pairs: {panel['symbol'].nunique()}")

    config = BacktestConfig(
        fee=args.fee,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
        start_capital_usdt=args.capital,
        use_triangle_only=not args.all_cycles,
        use_intrabar_hilo=args.intrabar_hilo,
    )

    print("Running backtest (Bellman-Ford negative cycles)...")
    bt = run_backtest(panel, config)
    summary = bt.summary()

    print("\n=== BACKTEST SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save outputs
    summary_path = results / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    bt.equity_curve.to_csv(results / "equity_curve.csv", header=["equity"])
    if bt.trades:
        trades_df = pd.DataFrame(
            [
                {
                    "time": t.time,
                    "signal_time": t.signal_time,
                    "cycle": " -> ".join(t.cycle),
                    "gross_mult": t.gross_multiplier,
                    "realized_gross_mult": t.realized_gross_multiplier,
                    "net_mult": t.net_multiplier,
                    "pnl_usdt": t.pnl_usdt,
                    "capital": t.capital_after,
                }
                for t in bt.trades
            ]
        )
        trades_df.to_csv(results / "trades.csv", index=False)

    if not bt.equity_curve.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        bt.equity_curve.plot(ax=ax, title="Equity curve (USDT, synthetic arb)")
        ax.set_ylabel("Capital (USDT)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results / "equity_curve.png", dpi=120)
        plt.close()

    print(f"\nResults saved to {results}/")


if __name__ == "__main__":
    main()
