#!/usr/bin/env python3
"""Real L2 order-book backtest on a free Tardis.dev sample (1 day, first-of-month).

Demonstrates that the cross-exchange arbitrage method was tested on actual
tick-level order book data (true bid/ask + depth), not only 5m OHLC closes.

    python3 run_tardis.py --date 2026-05-01 --assets BTC ETH

Data: free first-day-of-month sample from datasets.tardis.dev (no API key).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.tardis_backtest import L2Config, run_l2_backtest
from src.tardis_data import load_l2_panels


TRADE_COLUMNS = [
    "signal_ts_ms", "exec_ts_ms", "asset", "buy_ex", "sell_ex", "size",
    "buy_cost", "sell_proceeds", "pnl", "signal_edge_bps", "signal_expected_pnl",
]


def write_trades_csv(trades, path: Path) -> None:
    pd.DataFrame([{
        "signal_ts_ms": t.ts, "exec_ts_ms": t.exec_ts,
        "asset": t.asset, "buy_ex": t.buy_ex, "sell_ex": t.sell_ex,
        "size": t.size, "buy_cost": t.buy_cost, "sell_proceeds": t.sell_proceeds,
        "pnl": t.pnl,
        "signal_edge_bps": t.signal_edge_bps,
        "signal_expected_pnl": t.signal_expected_pnl,
    } for t in trades], columns=TRADE_COLUMNS).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-01", help="First day of a month (free tier)")
    ap.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--exchanges", nargs="+", default=["binance", "bybit", "okx"])
    ap.add_argument("--notional", type=float, default=2_000.0)
    ap.add_argument("--min-profit-usdt", type=float, default=0.01,
                    help="Minimum expected PnL at signal time required to send a trade")
    ap.add_argument("--min-edge-bps", type=float, default=0.0,
                    help="Minimum net edge after fees, in basis points")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--grid-ms", type=int, default=100,
                    help="Time-grid resolution in ms")
    ap.add_argument("--max-quote-age-ms", type=int, default=250,
                    help="Drop forward-filled books older than this many ms")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    raw_dir = root / "data" / "tardis" / "raw"
    grid_dir = root / "data" / "tardis" / "grid"
    results = root / "results"
    results.mkdir(exist_ok=True)

    print(f"Loading Tardis L2 panels ({args.date}, depth={args.depth}, grid={args.grid_ms}ms)...")
    panels = load_l2_panels(
        assets=args.assets,
        exchanges=args.exchanges,
        date=args.date,
        raw_dir=raw_dir,
        grid_dir=grid_dir,
        depth=args.depth,
        grid_ms=args.grid_ms,
    )
    if len(panels) < 2:
        raise SystemExit("Need at least two (exchange, asset) panels.")

    venues = sorted({ex for ex, _ in panels})
    assets = sorted({a for _, a in panels})
    print(f"  venues={venues} assets={assets}")

    def cfg(fee: float, lat_ms: int) -> L2Config:
        return L2Config(fee=fee, max_notional_usdt=args.notional,
                        min_profit_usdt=args.min_profit_usdt,
                        min_edge_bps=args.min_edge_bps,
                        depth=args.depth,
                        latency_ms=lat_ms, grid_ms=args.grid_ms,
                        max_quote_age_ms=args.max_quote_age_ms)

    # Two experiments:
    #  (1) NO DELAY (latency 0) — best case, isolates the effect of fees.
    #  (2) WITH DELAY — execution shifted to t+Δ on the real book; Δ swept to
    #      show how a realistic latency erodes the edge.
    scenarios = [
        # retail taker fee: no delay vs delay (fees alone should kill it)
        ("l2_taker_0p10_lat0",   "Тейкер 0.10%, БЕЗ задержки",        cfg(0.0010, 0)),
        ("l2_taker_0p10_lat250", "Тейкер 0.10%, задержка 250мс",      cfg(0.0010, 250)),
        # near-zero fee (HFT/maker tier): latency sweep shows the decay
        ("l2_fee_0p01_lat0",     "Комиссия 0.01%, БЕЗ задержки",      cfg(0.0001, 0)),
        ("l2_fee_0p01_lat100",   "Комиссия 0.01%, задержка 100мс",    cfg(0.0001, 100)),
        ("l2_fee_0p01_lat250",   "Комиссия 0.01%, задержка 250мс",    cfg(0.0001, 250)),
        ("l2_fee_0p01_lat300",   "Комиссия 0.01%, задержка 300мс",    cfg(0.0001, 300)),
        ("l2_fee_0p01_lat500",   "Комиссия 0.01%, задержка 500мс",    cfg(0.0001, 500)),
        # zero-fee, no-delay ceiling
        ("l2_no_fees_lat0",      "Без комиссий, БЕЗ задержки (потолок)", cfg(0.0, 0)),
    ]

    summaries = []
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for key, title, cfg in scenarios:
        print(f"\nRunning: {title}")
        res = run_l2_backtest(panels, cfg)
        s = res.summary()
        s["scenario"] = title
        s["fee_per_leg"] = cfg.fee
        s["latency_ms"] = cfg.latency_ms
        s["max_quote_age_ms"] = cfg.max_quote_age_ms
        summaries.append(s)
        print(f"  trades={s['trades_executed']}  pnl={s['total_pnl_usdt']:.2f} USDT  "
              f"return={s['total_return_pct']:.3f}%  raw_crosses={s['raw_cross_candidates']}  "
              f"exec_candidates={s['executable_candidates']}")
        print(f"  by_route={s['by_route']}")

        res.equity_curve.to_csv(results / f"tardis_equity_{key}.csv", header=["equity"])
        write_trades_csv(res.trades, results / f"tardis_trades_{key}.csv")
        if not res.equity_curve.empty:
            res.equity_curve.plot(ax=ax, lw=1.1, label=title)

    ax.set_title(f"L2 cross-exchange arbitrage — Tardis {args.date} ({'/'.join(venues)})")
    ax.set_ylabel("Capital (USDT)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results / "tardis_equity.png", dpi=120)
    plt.close()

    meta = {
        "data_source": "Tardis.dev free first-of-month L2 sample (book_snapshot_25)",
        "date": args.date,
        "venues": venues,
        "assets": assets,
        "max_notional_usdt": args.notional,
        "min_profit_usdt": args.min_profit_usdt,
        "min_edge_bps": args.min_edge_bps,
        "book_depth_levels": args.depth,
        "grid_ms": args.grid_ms,
        "max_quote_age_ms": args.max_quote_age_ms,
        "model": "pre-funded inventory, slippage via order-book walking, "
                 "latency-shifted execution (decide at t, fill on first grid point "
                 "at/after t+Δ), spot buy fee deducted from base asset, spot sell fee "
                 "deducted from quote proceeds, stale forward-filled books capped by "
                 "max_quote_age_ms",
        "scenarios": summaries,
    }
    with open(results / "tardis_l2_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {results}/tardis_l2_report.json, tardis_equity.png")


if __name__ == "__main__":
    main()
