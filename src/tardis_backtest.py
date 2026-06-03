"""Event-driven cross-exchange arbitrage backtest on real Tardis.dev L2 data.

Differences from the OHLC backtest (src/cross_exchange.py):

* Uses REAL order book (top-N bid/ask levels), not 5m close as a mid proxy.
* Slippage is computed by WALKING the book for the traded size, not a fixed bps.
* Latency-aware at the MILLISECOND scale: decide on the book at t, execute on the
  book as it actually was at t+latency_ms (latency-shifted execution on real
  timestamps — no arbitrary "window" cutting).
* Pre-funded INVENTORY model: assets sit on both venues, so there is no
  coin-transfer inside the trade (the realistic way cross-exchange arb is run);
  rebalancing is an offline cost, not part of per-trade PnL.

This is the "tested on real L2 data" demonstration, on one free day
(first-of-month Tardis sample).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class L2Config:
    fee: float = 0.001           # taker fee per leg (0.1%)
    max_notional_usdt: float = 2_000.0   # per-opportunity size cap
    min_profit_usdt: float = 0.01        # ignore sub-cent edges
    min_edge_bps: float = 0.0    # require net edge after fees, in basis points
    latency_ms: int = 0          # execution delay vs. signal (milliseconds)
    grid_ms: int = 100           # time-grid resolution (must match loaded panels)
    depth: int = 5               # book levels used for slippage
    start_capital_usdt: float = 10_000.0
    max_quote_age_ms: int = 250  # cap stale forward-filled books


@dataclass
class L2Trade:
    ts: int                      # milliseconds
    exec_ts: int                 # milliseconds
    asset: str
    buy_ex: str
    sell_ex: str
    size: float
    buy_cost: float
    sell_proceeds: float
    pnl: float
    signal_edge_bps: float
    signal_expected_pnl: float


@dataclass
class L2Result:
    trades: List[L2Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    grid_points: int = 0
    raw_crosses: int = 0
    executable_candidates: int = 0
    start_capital_usdt: float = 10_000.0

    def summary(self) -> Dict:
        pnl = sum(t.pnl for t in self.trades)
        cap0 = self.start_capital_usdt
        cap1 = cap0 + pnl
        return {
            "grid_points": self.grid_points,
            "raw_cross_candidates": self.raw_crosses,
            "executable_candidates": self.executable_candidates,
            "trades_executed": len(self.trades),
            "total_pnl_usdt": float(pnl),
            "total_return_pct": float((cap1 / cap0 - 1) * 100) if cap0 else 0.0,
            "final_capital": float(cap1),
            "avg_trade_pnl": float(pnl / len(self.trades)) if self.trades else 0.0,
            "by_asset": _pnl_by_key(self.trades, lambda t: t.asset),
            "by_route": _pnl_by_key(self.trades, lambda t: f"{t.buy_ex}->{t.sell_ex}"),
        }


def _pnl_by_key(trades: List[L2Trade], key) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in trades:
        k = key(t)
        d = out.setdefault(k, {"trades": 0, "pnl": 0.0})
        d["trades"] += 1
        d["pnl"] += t.pnl
    return {k: {"trades": v["trades"], "pnl": round(v["pnl"], 4)} for k, v in out.items()}


def _walk_buy(row: dict, size: float, depth: int) -> Tuple[float, float]:
    """Buy up to `size` asset units by consuming ask levels. Returns (filled, cost)."""
    remaining, cost = size, 0.0
    for i in range(depth):
        p = row.get(f"asks[{i}].price")
        a = row.get(f"asks[{i}].amount")
        if p is None or a is None or not np.isfinite(p) or not np.isfinite(a) or a <= 0:
            continue
        take = min(remaining, a)
        cost += take * p
        remaining -= take
        if remaining <= 1e-15:
            break
    return size - remaining, cost


def _walk_sell(row: dict, size: float, depth: int) -> Tuple[float, float]:
    """Sell up to `size` asset units into bid levels. Returns (filled, proceeds)."""
    remaining, proceeds = size, 0.0
    for i in range(depth):
        p = row.get(f"bids[{i}].price")
        a = row.get(f"bids[{i}].amount")
        if p is None or a is None or not np.isfinite(p) or not np.isfinite(a) or a <= 0:
            continue
        take = min(remaining, a)
        proceeds += take * p
        remaining -= take
        if remaining <= 1e-15:
            break
    return size - remaining, proceeds


def _depth_sum(row: dict, side: str, depth: int) -> float:
    s = 0.0
    for i in range(depth):
        a = row.get(f"{side}[{i}].amount")
        if a is not None and np.isfinite(a) and a > 0:
            s += a
    return s


def _align_asset(
    panels: Dict[Tuple[str, str], pd.DataFrame], asset: str
) -> Tuple[List[str], List[int], Dict[str, dict]]:
    """Forward-fill each venue's book onto the union time grid for one asset.

    NOTE: ffill across venues means a venue that updated a few ticks ago is
    compared against a fresher one — non-contemporaneous quotes can look like a
    cross. This inflates the raw-candidate count; latency-shifted execution and
    fees filter most of it out, but it is a known limitation of grid alignment.
    """
    venues = sorted(ex for (ex, a) in panels if a == asset)
    if len(venues) < 2:
        return venues, [], {}
    tss = sorted(set().union(*[set(panels[(ex, asset)]["ts"]) for ex in venues]))
    idx = pd.Index(tss, name="ts")
    aligned: Dict[str, dict] = {}
    for ex in venues:
        df = panels[(ex, asset)].copy()
        df["source_ts"] = df["ts"]
        df = df.set_index("ts").reindex(idx).ffill()
        aligned[ex] = df.to_dict("index")
    return venues, tss, aligned


def _is_fresh(row: dict, t: int, max_age_ms: int) -> bool:
    source_ts = row.get("source_ts")
    if source_ts is None or not np.isfinite(source_ts):
        return False
    return 0 <= t - int(source_ts) <= max_age_ms


def run_l2_backtest(
    panels: Dict[Tuple[str, str], pd.DataFrame],
    config: L2Config,
) -> L2Result:
    assets = sorted({a for (_, a) in panels})
    trades: List[L2Trade] = []
    raw_crosses = 0
    executable_candidates = 0

    for asset in assets:
        venues, tss, aligned = _align_asset(panels, asset)
        if len(venues) < 2:
            continue
        depth = config.depth

        for n, t in enumerate(tss):
            # signal time t, execution on the first grid point at or after t+latency
            te_idx = bisect_left(tss, t + config.latency_ms, lo=n)
            if te_idx >= len(tss):
                break
            te = tss[te_idx]

            # pick buy venue (lowest ask) / sell venue (highest bid) at SIGNAL time
            best_ask_ex, best_ask = None, float("inf")
            best_bid_ex, best_bid = None, float("-inf")
            for ex in venues:
                r = aligned[ex].get(t)
                if r is None or not _is_fresh(r, t, config.max_quote_age_ms):
                    continue
                a0 = r.get("asks[0].price")
                b0 = r.get("bids[0].price")
                if a0 is not None and np.isfinite(a0) and a0 < best_ask:
                    best_ask, best_ask_ex = a0, ex
                if b0 is not None and np.isfinite(b0) and b0 > best_bid:
                    best_bid, best_bid_ex = b0, ex

            if best_ask_ex is None or best_bid_ex is None or best_ask_ex == best_bid_ex:
                continue
            if best_bid <= best_ask:  # no raw cross before costs
                continue
            raw_crosses += 1
            signal_gross_edge = (best_bid / best_ask) * (1.0 - config.fee) ** 2 - 1.0
            signal_edge_bps = signal_gross_edge * 10_000
            signal_expected_pnl = config.max_notional_usdt * signal_gross_edge
            if (
                signal_expected_pnl < config.min_profit_usdt
                or signal_edge_bps < config.min_edge_bps
            ):
                continue

            # execute on the books as they ACTUALLY were at t+latency
            buy_row = aligned[best_ask_ex].get(te)
            sell_row = aligned[best_bid_ex].get(te)
            if buy_row is None or sell_row is None:
                continue
            if not _is_fresh(buy_row, te, config.max_quote_age_ms):
                continue
            if not _is_fresh(sell_row, te, config.max_quote_age_ms):
                continue
            exec_ask = buy_row.get("asks[0].price")
            if exec_ask is None or not np.isfinite(exec_ask) or exec_ask <= 0:
                continue

            target_qty = config.max_notional_usdt / exec_ask
            size = min(
                target_qty,
                _depth_sum(buy_row, "asks", depth),
                _depth_sum(sell_row, "bids", depth),
            )
            if size <= 0:
                continue

            f_buy, _ = _walk_buy(buy_row, size, depth)
            f_sell, _ = _walk_sell(sell_row, size, depth)
            size = min(f_buy, f_sell)
            if size <= 0:
                continue
            _, buy_cost = _walk_buy(buy_row, size, depth)
            _, sell_proceeds = _walk_sell(sell_row, size, depth)

            buy_cost *= (1.0 + config.fee)
            sell_proceeds *= (1.0 - config.fee)
            pnl = sell_proceeds - buy_cost

            executable_candidates += 1
            trades.append(
                L2Trade(
                    ts=t, exec_ts=te, asset=asset,
                    buy_ex=best_ask_ex, sell_ex=best_bid_ex,
                    size=size, buy_cost=buy_cost,
                    sell_proceeds=sell_proceeds, pnl=pnl,
                    signal_edge_bps=signal_edge_bps,
                    signal_expected_pnl=signal_expected_pnl,
                )
            )

    # Equity curve built from CHRONOLOGICALLY SORTED trades across all assets,
    # so the curve is time-consistent regardless of per-asset processing order.
    all_ts = sorted(set().union(*[set(df["ts"]) for df in panels.values()]))
    pnl_at = pd.Series(0.0, index=all_ts)
    for tr in trades:
        if tr.exec_ts in pnl_at.index:
            pnl_at.loc[tr.exec_ts] += tr.pnl
    equity = config.start_capital_usdt + pnl_at.cumsum()
    equity.index = pd.to_datetime(all_ts, unit="ms", utc=True)

    return L2Result(
        trades=sorted(trades, key=lambda x: (x.exec_ts, x.ts)),
        equity_curve=equity,
        grid_points=len(all_ts),
        raw_crosses=raw_crosses,
        executable_candidates=executable_candidates,
        start_capital_usdt=config.start_capital_usdt,
    )
