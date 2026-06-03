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
import os
import sys
from pathlib import Path

import pandas as pd

from src.tardis_backtest import L2Config, run_l2_backtest, run_l2_combined_backtest
from src.tardis_data import load_l2_pair_panels, load_l2_panels, normalize_pair


DEFAULT_FEES_PCT = [0.0, 0.01, 0.1]
DEFAULT_LATENCIES_MS = [0, 100, 300, 500]

TRADE_COLUMNS = [
    "signal_ts_ms", "exec_ts_ms", "asset", "buy_ex", "sell_ex", "size",
    "buy_cost", "sell_proceeds", "pnl", "signal_edge_bps", "signal_expected_pnl",
]

TRIANGULAR_TRADE_COLUMNS = [
    "signal_ts_ms", "exec_ts_ms", "exchange", "cycle", "start_currency",
    "start_amount", "end_amount", "start_value_usdt", "end_value_usdt",
    "pnl", "signal_edge_bps", "signal_expected_pnl",
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


def write_triangular_trades_csv(trades, path: Path) -> None:
    pd.DataFrame([{
        "signal_ts_ms": t.ts, "exec_ts_ms": t.exec_ts,
        "exchange": t.exchange, "cycle": "->".join(t.cycle),
        "start_currency": t.start_currency,
        "start_amount": t.start_amount, "end_amount": t.end_amount,
        "start_value_usdt": t.start_value_usdt,
        "end_value_usdt": t.end_value_usdt,
        "pnl": t.pnl,
        "signal_edge_bps": t.signal_edge_bps,
        "signal_expected_pnl": t.signal_expected_pnl,
    } for t in trades], columns=TRIANGULAR_TRADE_COLUMNS).to_csv(path, index=False)


def _dedupe_preserve_order(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _number_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def scenario_key(date: str, fee_pct: float, latency_ms: int) -> str:
    date_tag = date.replace("-", "")
    return f"{date_tag}_fee{_number_tag(fee_pct)}pct_lat{latency_ms}"


def scenario_title(fee_pct: float, latency_ms: int) -> str:
    fee_label = "Без комиссий" if fee_pct == 0 else f"Комиссия {fee_pct:g}%"
    latency_label = "без задержки" if latency_ms == 0 else f"задержка {latency_ms}мс"
    return f"{fee_label}, {latency_label}"


def build_scenarios(date: str, fees_pct, latencies_ms, cfg_factory):
    scenarios = []
    for fee_pct in _dedupe_preserve_order(fees_pct):
        if fee_pct < 0:
            raise ValueError(f"Fee percent must be non-negative: {fee_pct}")
        for latency_ms in _dedupe_preserve_order(latencies_ms):
            if latency_ms < 0:
                raise ValueError(f"Latency must be non-negative: {latency_ms}")
            fee = fee_pct / 100.0
            scenarios.append(
                (
                    scenario_key(date, fee_pct, latency_ms),
                    scenario_title(fee_pct, latency_ms),
                    fee_pct,
                    cfg_factory(fee, latency_ms),
                )
            )
    return scenarios


def triangle_pairs_from_assets(assets):
    ordered_assets = _dedupe_preserve_order([a.upper() for a in assets])
    pairs = [(asset, "USDT") for asset in ordered_assets]
    for quote_index, quote in enumerate(ordered_assets):
        for base in ordered_assets[quote_index + 1:]:
            pairs.append((base, quote))
    return pairs


def parse_pairs(pair_args):
    return _dedupe_preserve_order([normalize_pair(pair) for pair in pair_args])


def configure_runtime(root: Path) -> None:
    """Keep long L2 runs chatty and avoid Matplotlib writing outside the repo."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    mpl_config = root / ".cache" / "matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-01", help="First day of a month (free tier)")
    ap.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--exchanges", nargs="+", default=["binance", "bybit", "okx"])
    ap.add_argument("--mode", choices=["direct", "direct-plus-triangles"], default="direct",
                    help="direct = cross-exchange cycles length 2; direct-plus-triangles "
                         "= unified direct + same-exchange length-3 strategy with shared inventory")
    ap.add_argument("--output-tag",
                    help="Optional suffix for result filenames, useful for parallel runs")
    ap.add_argument("--triangle-pairs", nargs="+",
                    help="Optional pair override for triangle mode, e.g. BTC/USDT ETH/USDT ETH/BTC")
    ap.add_argument("--notional", type=float,
                    help="Optional USDT cap per trade; omitted means size only by stake percent")
    ap.add_argument("--stake-pct", type=float, default=20.0,
                    help="Maximum share of available start inventory to use per opportunity")
    ap.add_argument("--min-profit-usdt", type=float, default=0.0,
                    help="Optional absolute expected-PnL floor at signal time")
    ap.add_argument("--min-profit-pct", type=float, default=0.5,
                    help="Minimum expected PnL at signal time as percent of notional")
    ap.add_argument("--min-edge-bps", type=float, default=0.0,
                    help="Minimum net edge after fees, in basis points")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--grid-ms", type=int, default=100,
                    help="Time-grid resolution in ms")
    ap.add_argument("--max-quote-age-ms", type=int, default=250,
                    help="Drop forward-filled books older than this many ms")
    ap.add_argument("--fees-pct", nargs="+", type=float, default=DEFAULT_FEES_PCT,
                    help="Spot taker fee per leg in percent, e.g. 0 0.01 0.1")
    ap.add_argument("--latencies-ms", nargs="+", type=int, default=DEFAULT_LATENCIES_MS,
                    help="Execution latencies to sweep in milliseconds")
    ap.add_argument("--inventory-per-currency-usdt", type=float, default=5_000.0,
                    help="Initial USDT-equivalent balance for each currency on each exchange")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    configure_runtime(root)
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
                        stake_fraction=args.stake_pct / 100.0,
                        min_profit_usdt=args.min_profit_usdt,
                        min_profit_pct=args.min_profit_pct,
                        min_edge_bps=args.min_edge_bps,
                        depth=args.depth,
                        latency_ms=lat_ms, grid_ms=args.grid_ms,
                        inventory_per_currency_usdt=args.inventory_per_currency_usdt,
                        max_quote_age_ms=args.max_quote_age_ms)

    scenarios = build_scenarios(args.date, args.fees_pct, args.latencies_ms, cfg)
    date_tag = args.date.replace("-", "")
    output_tag = f"_{args.output_tag}" if args.output_tag else ""

    import matplotlib.pyplot as plt

    if args.mode == "direct":
        summaries = []
        fig, ax = plt.subplots(figsize=(10, 4))
        for key, title, fee_pct, cfg in scenarios:
            print(f"\nRunning: {title}")
            res = run_l2_backtest(panels, cfg)
            s = res.summary()
            s["scenario"] = title
            s["fee_pct"] = fee_pct
            s["fee_per_leg"] = cfg.fee
            s["latency_ms"] = cfg.latency_ms
            s["max_quote_age_ms"] = cfg.max_quote_age_ms
            summaries.append(s)
            print(f"  trades={s['trades_executed']}  pnl={s['total_pnl_usdt']:.2f} USDT  "
                  f"return={s['total_return_pct']:.3f}%  raw_crosses={s['raw_cross_candidates']}  "
                  f"exec_candidates={s['executable_candidates']}  "
                  f"inventory_skips={s['inventory_skips']}")
            print(f"  by_route={s['by_route']}")

            res.equity_curve.to_csv(results / f"tardis_equity_{key}{output_tag}.csv", header=["equity"])
            write_trades_csv(res.trades, results / f"tardis_trades_{key}{output_tag}.csv")
            if not res.equity_curve.empty:
                res.equity_curve.plot(ax=ax, lw=1.1, label=title)

        ax.set_title(f"L2 cross-exchange arbitrage — Tardis {args.date} ({'/'.join(venues)})")
        ax.set_ylabel("Capital (USDT)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dated_plot_path = results / f"tardis_equity_{date_tag}{output_tag}.png"
        latest_plot_path = results / f"tardis_equity{output_tag}.png"
        fig.savefig(dated_plot_path, dpi=120)
        fig.savefig(latest_plot_path, dpi=120)
        plt.close()

        meta = {
            "data_source": "Tardis.dev free first-of-month L2 sample (book_snapshot_25)",
            "date": args.date,
            "venues": venues,
            "assets": assets,
            "max_notional_usdt": args.notional,
            "stake_pct": args.stake_pct,
            "inventory_per_currency_usdt": args.inventory_per_currency_usdt,
            "min_profit_usdt": args.min_profit_usdt,
            "min_profit_pct": args.min_profit_pct,
            "min_edge_bps": args.min_edge_bps,
            "book_depth_levels": args.depth,
            "grid_ms": args.grid_ms,
            "max_quote_age_ms": args.max_quote_age_ms,
            "fees_pct": args.fees_pct,
            "latencies_ms": args.latencies_ms,
            "model": "pre-funded inventory, slippage via order-book walking, "
                     "latency-shifted execution (decide at t, fill on first grid point "
                     "at/after t+Δ), spot buy fee deducted from base asset, spot sell fee "
                     "deducted from quote proceeds, stale forward-filled books capped by "
                     "max_quote_age_ms",
            "scenarios": summaries,
        }
        dated_report_path = results / f"tardis_l2_report_{date_tag}{output_tag}.json"
        latest_report_path = results / f"tardis_l2_report{output_tag}.json"
        for path in (dated_report_path, latest_report_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {dated_report_path}, {dated_plot_path}")
        print(f"Latest aliases: {latest_report_path}, {latest_plot_path}")
        return

    triangle_pairs = (
        parse_pairs(args.triangle_pairs)
        if args.triangle_pairs
        else triangle_pairs_from_assets(args.assets)
    )
    print(
        "\nLoading Tardis L2 triangle panels "
        f"({args.date}, pairs={' '.join(f'{b}/{q}' for b, q in triangle_pairs)})..."
    )
    pair_panels = load_l2_pair_panels(
        pairs=triangle_pairs,
        exchanges=args.exchanges,
        date=args.date,
        raw_dir=raw_dir,
        grid_dir=grid_dir,
        depth=args.depth,
        grid_ms=args.grid_ms,
    )
    if len(pair_panels) < 3:
        raise SystemExit("Need at least three available pair panels for combined mode.")

    pair_labels = [f"{base}/{quote}" for base, quote in triangle_pairs]
    combined_summaries = []
    triangle_path_cache = {}
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, title, fee_pct, cfg in scenarios:
        print(f"\nRunning combined direct+triangles: {title}")
        res = run_l2_combined_backtest(
            panels,
            pair_panels,
            cfg,
            triangle_path_cache=triangle_path_cache,
        )
        s = res.summary()
        s["scenario"] = title
        s["fee_pct"] = fee_pct
        s["fee_per_leg"] = cfg.fee
        s["latency_ms"] = cfg.latency_ms
        s["max_quote_age_ms"] = cfg.max_quote_age_ms
        combined_summaries.append(s)
        print(f"  trades={s['trades_executed']}  direct={s['direct_trades_executed']}  "
              f"triangles={s['triangular_trades_executed']}  "
              f"pnl={s['total_pnl_usdt']:.2f} USDT  return={s['total_return_pct']:.3f}%  "
              f"raw_crosses={s['raw_cross_candidates']}  raw_cycles={s['raw_cycles']}  "
              f"exec_candidates={s['executable_candidates']}  "
              f"inventory_skips={s['inventory_skips']}")
        print(f"  by_type={s['by_type']}")
        print(f"  by_route={s['by_route']}")
        print(f"  by_cycle={s['by_cycle']}")

        res.equity_curve.to_csv(
            results / f"tardis_combined_equity_{key}{output_tag}.csv",
            header=["equity"],
        )
        write_trades_csv(
            res.direct_trades,
            results / f"tardis_combined_direct_trades_{key}{output_tag}.csv",
        )
        write_triangular_trades_csv(
            res.triangular_trades,
            results / f"tardis_combined_triangle_trades_{key}{output_tag}.csv",
        )
        if not res.equity_curve.empty:
            res.equity_curve.plot(ax=ax, lw=1.1, label=title)

    ax.set_title(f"L2 combined direct + triangles — Tardis {args.date}")
    ax.set_ylabel("Capital (USDT)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    dated_plot_path = results / f"tardis_combined_equity_{date_tag}{output_tag}.png"
    latest_plot_path = results / f"tardis_combined_equity{output_tag}.png"
    fig.savefig(dated_plot_path, dpi=120)
    fig.savefig(latest_plot_path, dpi=120)
    plt.close(fig)

    meta = {
        "data_source": "Tardis.dev free first-of-month L2 sample (book_snapshot_25)",
        "date": args.date,
        "venues": args.exchanges,
        "assets": assets,
        "pairs": pair_labels,
        "max_notional_usdt": args.notional,
        "stake_pct": args.stake_pct,
        "inventory_per_currency_usdt": args.inventory_per_currency_usdt,
        "min_profit_usdt": args.min_profit_usdt,
        "min_profit_pct": args.min_profit_pct,
        "min_edge_bps": args.min_edge_bps,
        "book_depth_levels": args.depth,
        "grid_ms": args.grid_ms,
        "max_quote_age_ms": args.max_quote_age_ms,
        "fees_pct": args.fees_pct,
        "latencies_ms": args.latencies_ms,
        "model": "unified pre-funded inventory strategy: at each signal timestamp, "
                 "cross-exchange direct candidates and same-exchange length-3 "
                 "triangular candidates compete for one shared (exchange, currency) "
                 "inventory; the highest expected-PnL signal is reserved and executed "
                 "on the first grid point at/after t+Δ; gross triangle path discovery "
                 "is cached across fee/latency scenarios",
        "scenarios": combined_summaries,
    }
    dated_report_path = results / f"tardis_combined_report_{date_tag}{output_tag}.json"
    latest_report_path = results / f"tardis_combined_report{output_tag}.json"
    for path in (dated_report_path, latest_report_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved combined: {dated_report_path}, {dated_plot_path}")
    print(f"Latest combined aliases: {latest_report_path}, {latest_plot_path}")


if __name__ == "__main__":
    main()
