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
* Spot fee convention: the buy-side fee reduces the asset received; the
  sell-side fee reduces the quote proceeds.

This is the "tested on real L2 data" demonstration, on one free day
(first-of-month Tardis sample).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .bellman_ford import ArbitrageCycle, enumerate_triangles
from .graph import ExchangeGraph


@dataclass
class L2Config:
    fee: float = 0.001           # taker fee per leg (0.1%)
    max_notional_usdt: float | None = None   # optional per-opportunity USDT cap
    stake_fraction: float | None = None      # None = fixed notional, otherwise inventory share
    min_profit_usdt: float = 0.0         # optional absolute expected-PnL floor
    min_profit_pct: float = 0.0          # expected PnL as percent of notional
    min_edge_bps: float = 0.0    # require net edge after fees, in basis points
    latency_ms: int = 0          # execution delay vs. signal (milliseconds)
    grid_ms: int = 100           # time-grid resolution (must match loaded panels)
    depth: int = 5               # book levels used for slippage
    start_capital_usdt: float = 10_000.0
    inventory_per_currency_usdt: float | None = None
    max_quote_age_ms: int = 250  # cap stale forward-filled books
    enforce_inventory: bool = True


@dataclass
class L2Trade:
    ts: int                      # milliseconds
    exec_ts: int                 # milliseconds
    asset: str
    buy_ex: str
    sell_ex: str
    size: float                      # net asset sold after buy-side fee
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
    inventory_skips: int = 0
    start_capital_usdt: float = 10_000.0

    def summary(self) -> Dict:
        pnl = sum(t.pnl for t in self.trades)
        cap0 = self.start_capital_usdt
        cap1 = cap0 + pnl
        return {
            "grid_points": self.grid_points,
            "raw_cross_candidates": self.raw_crosses,
            "executable_candidates": self.executable_candidates,
            "inventory_skips": self.inventory_skips,
            "trades_executed": len(self.trades),
            "total_pnl_usdt": float(pnl),
            "total_return_pct": float((cap1 / cap0 - 1) * 100) if cap0 else 0.0,
            "final_capital": float(cap1),
            "avg_trade_pnl": float(pnl / len(self.trades)) if self.trades else 0.0,
            "by_asset": _pnl_by_key(self.trades, lambda t: t.asset),
            "by_route": _pnl_by_key(self.trades, lambda t: f"{t.buy_ex}->{t.sell_ex}"),
        }


@dataclass
class L2TriangularTrade:
    ts: int
    exec_ts: int
    exchange: str
    cycle: List[str]
    start_currency: str
    start_amount: float
    end_amount: float
    start_value_usdt: float
    end_value_usdt: float
    pnl: float
    signal_edge_bps: float
    signal_expected_pnl: float


@dataclass
class L2TriangularResult:
    trades: List[L2TriangularTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    grid_points: int = 0
    raw_cycles: int = 0
    executable_candidates: int = 0
    inventory_skips: int = 0
    start_capital_usdt: float = 10_000.0

    def summary(self) -> Dict:
        pnl = sum(t.pnl for t in self.trades)
        cap0 = self.start_capital_usdt
        cap1 = cap0 + pnl
        return {
            "grid_points": self.grid_points,
            "raw_cross_candidates": self.raw_cycles,
            "raw_cycles": self.raw_cycles,
            "executable_candidates": self.executable_candidates,
            "inventory_skips": self.inventory_skips,
            "trades_executed": len(self.trades),
            "total_pnl_usdt": float(pnl),
            "total_return_pct": float((cap1 / cap0 - 1) * 100) if cap0 else 0.0,
            "final_capital": float(cap1),
            "avg_trade_pnl": float(pnl / len(self.trades)) if self.trades else 0.0,
            "by_exchange": _tri_pnl_by_key(self.trades, lambda t: t.exchange),
            "by_cycle": _tri_pnl_by_key(self.trades, lambda t: "->".join(t.cycle)),
        }


@dataclass
class L2CombinedResult:
    direct_trades: List[L2Trade] = field(default_factory=list)
    triangular_trades: List[L2TriangularTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    grid_points: int = 0
    raw_crosses: int = 0
    raw_cycles: int = 0
    executable_candidates: int = 0
    inventory_skips: int = 0
    start_capital_usdt: float = 10_000.0

    def summary(self) -> Dict:
        direct_pnl = sum(t.pnl for t in self.direct_trades)
        triangular_pnl = sum(t.pnl for t in self.triangular_trades)
        pnl = direct_pnl + triangular_pnl
        cap0 = self.start_capital_usdt
        cap1 = cap0 + pnl
        return {
            "grid_points": self.grid_points,
            "raw_cross_candidates": self.raw_crosses,
            "raw_cycles": self.raw_cycles,
            "executable_candidates": self.executable_candidates,
            "inventory_skips": self.inventory_skips,
            "trades_executed": len(self.direct_trades) + len(self.triangular_trades),
            "direct_trades_executed": len(self.direct_trades),
            "triangular_trades_executed": len(self.triangular_trades),
            "total_pnl_usdt": float(pnl),
            "direct_pnl_usdt": float(direct_pnl),
            "triangular_pnl_usdt": float(triangular_pnl),
            "total_return_pct": float((cap1 / cap0 - 1) * 100) if cap0 else 0.0,
            "final_capital": float(cap1),
            "avg_trade_pnl": (
                float(pnl / (len(self.direct_trades) + len(self.triangular_trades)))
                if self.direct_trades or self.triangular_trades
                else 0.0
            ),
            "by_type": {
                "direct": {"trades": len(self.direct_trades), "pnl": round(direct_pnl, 4)},
                "triangle": {
                    "trades": len(self.triangular_trades),
                    "pnl": round(triangular_pnl, 4),
                },
            },
            "by_route": _pnl_by_key(
                self.direct_trades,
                lambda t: f"{t.buy_ex}->{t.sell_ex}",
            ),
            "by_cycle": _tri_pnl_by_key(
                self.triangular_trades,
                lambda t: "->".join(t.cycle),
            ),
        }


def _pnl_by_key(trades: List[L2Trade], key) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in trades:
        k = key(t)
        d = out.setdefault(k, {"trades": 0, "pnl": 0.0})
        d["trades"] += 1
        d["pnl"] += t.pnl
    return {k: {"trades": v["trades"], "pnl": round(v["pnl"], 4)} for k, v in out.items()}


def _tri_pnl_by_key(trades: List[L2TriangularTrade], key) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in trades:
        k = key(t)
        d = out.setdefault(k, {"trades": 0, "pnl": 0.0})
        d["trades"] += 1
        d["pnl"] += t.pnl
    return {k: {"trades": v["trades"], "pnl": round(v["pnl"], 4)} for k, v in out.items()}


def _min_signal_profit_usdt(config: L2Config, stake_value_usdt: float) -> float:
    pct_floor = stake_value_usdt * (config.min_profit_pct / 100.0)
    return max(config.min_profit_usdt, pct_floor)


def _min_signal_edge_bps(config: L2Config) -> float:
    pct_floor_bps = config.min_profit_pct * 100.0
    return max(config.min_edge_bps, pct_floor_bps)


def _apply_notional_cap(amount: float, value_usdt: float, config: L2Config) -> tuple[float, float]:
    if config.max_notional_usdt is None or value_usdt <= config.max_notional_usdt:
        return amount, value_usdt
    if value_usdt <= 0:
        return 0.0, 0.0
    scale = config.max_notional_usdt / value_usdt
    return amount * scale, config.max_notional_usdt


def _stake_fraction(config: L2Config) -> float:
    return 1.0 if config.stake_fraction is None else config.stake_fraction


def _fixed_or_fractional_notional(config: L2Config) -> float:
    if config.stake_fraction is None:
        return (
            config.max_notional_usdt
            if config.max_notional_usdt is not None
            else config.start_capital_usdt
        )
    return config.start_capital_usdt * config.stake_fraction


def _canonical_cycle_key(cycle: ArbitrageCycle) -> tuple[str, ...]:
    nodes = cycle.currencies[:-1]
    rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes))]
    return min(rotations)


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


@dataclass(frozen=True)
class _AssetView:
    venues: List[str]
    tss: List[int]
    tss_set: set[int]
    aligned: Dict[str, dict]


@dataclass(frozen=True)
class _PendingTrade:
    seq: int
    ts: int
    exec_ts: int
    asset: str
    buy_ex: str
    sell_ex: str
    reserved_quote: float
    reserved_asset: float
    signal_edge_bps: float
    signal_expected_pnl: float


@dataclass(frozen=True)
class _PairAction:
    pair: str
    side: str


@dataclass(frozen=True)
class _ExchangePairView:
    exchange: str
    pairs: List[str]
    tss: List[int]
    tss_set: set[int]
    aligned: Dict[str, dict]


@dataclass(frozen=True)
class _PendingTriangularTrade:
    seq: int
    ts: int
    exec_ts: int
    exchange: str
    cycle: List[str]
    reserved_currency: str
    reserved_amount: float
    signal_start_value_usdt: float
    signal_edge_bps: float
    signal_expected_pnl: float


@dataclass(frozen=True)
class _CombinedSignal:
    kind: str
    exec_ts: int
    expected_pnl: float
    edge_bps: float
    asset: str | None = None
    buy_ex: str | None = None
    sell_ex: str | None = None
    reserved_quote: float = 0.0
    reserved_asset: float = 0.0
    exchange: str | None = None
    cycle: List[str] | None = None
    reserved_currency: str | None = None
    reserved_amount: float = 0.0
    start_value_usdt: float = 0.0


def _initial_mid(aligned: Dict[str, dict], tss: List[int], venue: str, max_age_ms: int) -> float:
    for t in tss:
        row = aligned[venue].get(t)
        if row is None or not _is_fresh(row, t, max_age_ms):
            continue
        ask = row.get("asks[0].price")
        bid = row.get("bids[0].price")
        if (
            ask is not None
            and bid is not None
            and np.isfinite(ask)
            and np.isfinite(bid)
            and ask > 0
            and bid > 0
        ):
            return (float(ask) + float(bid)) / 2.0
    return 0.0


def _initial_balances(
    asset_views: Dict[str, _AssetView],
    config: L2Config,
) -> tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
    venues = sorted({venue for view in asset_views.values() for venue in view.venues})
    assets = sorted(asset_views)
    quote_balances = {venue: 0.0 for venue in venues}
    asset_balances = {(venue, asset): 0.0 for venue in venues for asset in assets}
    if not venues or not assets:
        return quote_balances, asset_balances, 0.0

    if config.inventory_per_currency_usdt is not None:
        per_currency = config.inventory_per_currency_usdt
        for venue in venues:
            quote_balances[venue] = per_currency
        for asset, view in asset_views.items():
            for venue in view.venues:
                mid = _initial_mid(view.aligned, view.tss, venue, config.max_quote_age_ms)
                if mid > 0:
                    asset_balances[(venue, asset)] = per_currency / mid
        start_capital = per_currency * len(venues) * (len(assets) + 1)
        return quote_balances, asset_balances, start_capital

    quote_budget = config.start_capital_usdt * 0.5
    asset_budget = config.start_capital_usdt - quote_budget
    quote_per_venue = quote_budget / len(venues)
    for venue in venues:
        quote_balances[venue] = quote_per_venue

    asset_budget_per_slot = asset_budget / (len(venues) * len(assets))
    for asset, view in asset_views.items():
        for venue in view.venues:
            mid = _initial_mid(view.aligned, view.tss, venue, config.max_quote_age_ms)
            if mid > 0:
                asset_balances[(venue, asset)] = asset_budget_per_slot / mid

    return quote_balances, asset_balances, config.start_capital_usdt


def _execute_pending_trade(
    pending: _PendingTrade,
    view: _AssetView,
    config: L2Config,
    quote_balances: Dict[str, float],
    asset_balances: Dict[Tuple[str, str], float],
) -> L2Trade | None:
    buy_row = view.aligned[pending.buy_ex].get(pending.exec_ts)
    sell_row = view.aligned[pending.sell_ex].get(pending.exec_ts)
    if buy_row is None or sell_row is None:
        return None
    if not _is_fresh(buy_row, pending.exec_ts, config.max_quote_age_ms):
        return None
    if not _is_fresh(sell_row, pending.exec_ts, config.max_quote_age_ms):
        return None

    exec_ask = buy_row.get("asks[0].price")
    if exec_ask is None or not np.isfinite(exec_ask) or exec_ask <= 0:
        return None

    fee_factor = 1.0 - config.fee
    if fee_factor <= 0:
        return None

    depth = config.depth
    quote_cap = pending.reserved_quote
    sell_asset_cap = pending.reserved_asset
    target_gross_qty = quote_cap / exec_ask
    gross_buy_size = min(
        target_gross_qty,
        _depth_sum(buy_row, "asks", depth),
        _depth_sum(sell_row, "bids", depth) / fee_factor,
        sell_asset_cap / fee_factor,
    )
    if gross_buy_size <= 0:
        return None

    f_buy, _ = _walk_buy(buy_row, gross_buy_size, depth)
    size = f_buy * fee_factor
    f_sell, _ = _walk_sell(sell_row, size, depth)
    size = min(size, f_sell, sell_asset_cap)
    if size <= 0:
        return None

    gross_buy_size = size / fee_factor
    _, buy_cost = _walk_buy(buy_row, gross_buy_size, depth)
    _, sell_proceeds = _walk_sell(sell_row, size, depth)
    sell_proceeds *= fee_factor
    pnl = sell_proceeds - buy_cost

    if config.enforce_inventory:
        quote_balances[pending.buy_ex] += max(0.0, pending.reserved_quote - buy_cost)
        asset_balances[(pending.buy_ex, pending.asset)] += size
        asset_balances[(pending.sell_ex, pending.asset)] += max(
            0.0,
            pending.reserved_asset - size,
        )
        quote_balances[pending.sell_ex] += sell_proceeds

    return L2Trade(
        ts=pending.ts,
        exec_ts=pending.exec_ts,
        asset=pending.asset,
        buy_ex=pending.buy_ex,
        sell_ex=pending.sell_ex,
        size=size,
        buy_cost=buy_cost,
        sell_proceeds=sell_proceeds,
        pnl=pnl,
        signal_edge_bps=pending.signal_edge_bps,
        signal_expected_pnl=pending.signal_expected_pnl,
    )


def _release_pending_reservation(
    pending: _PendingTrade,
    quote_balances: Dict[str, float],
    asset_balances: Dict[Tuple[str, str], float],
) -> None:
    quote_balances[pending.buy_ex] += pending.reserved_quote
    asset_balances[(pending.sell_ex, pending.asset)] += pending.reserved_asset


def _release_direct_currency_reservation(
    pending: _PendingTrade,
    currency_balances: Dict[Tuple[str, str], float],
) -> None:
    currency_balances[(pending.buy_ex, "USDT")] += pending.reserved_quote
    currency_balances[(pending.sell_ex, pending.asset)] += pending.reserved_asset


def _execute_pending_direct_currency(
    pending: _PendingTrade,
    view: _AssetView,
    config: L2Config,
    currency_balances: Dict[Tuple[str, str], float],
) -> L2Trade | None:
    buy_row = view.aligned[pending.buy_ex].get(pending.exec_ts)
    sell_row = view.aligned[pending.sell_ex].get(pending.exec_ts)
    if buy_row is None or sell_row is None:
        return None
    if not _is_fresh(buy_row, pending.exec_ts, config.max_quote_age_ms):
        return None
    if not _is_fresh(sell_row, pending.exec_ts, config.max_quote_age_ms):
        return None

    exec_ask = buy_row.get("asks[0].price")
    if exec_ask is None or not np.isfinite(exec_ask) or exec_ask <= 0:
        return None

    fee_factor = 1.0 - config.fee
    if fee_factor <= 0:
        return None

    depth = config.depth
    quote_cap = pending.reserved_quote
    sell_asset_cap = pending.reserved_asset
    target_gross_qty = quote_cap / exec_ask
    gross_buy_size = min(
        target_gross_qty,
        _depth_sum(buy_row, "asks", depth),
        _depth_sum(sell_row, "bids", depth) / fee_factor,
        sell_asset_cap / fee_factor,
    )
    if gross_buy_size <= 0:
        return None

    f_buy, _ = _walk_buy(buy_row, gross_buy_size, depth)
    size = f_buy * fee_factor
    f_sell, _ = _walk_sell(sell_row, size, depth)
    size = min(size, f_sell, sell_asset_cap)
    if size <= 0:
        return None

    gross_buy_size = size / fee_factor
    _, buy_cost = _walk_buy(buy_row, gross_buy_size, depth)
    _, sell_proceeds = _walk_sell(sell_row, size, depth)
    sell_proceeds *= fee_factor
    pnl = sell_proceeds - buy_cost

    if config.enforce_inventory:
        currency_balances[(pending.buy_ex, "USDT")] += max(
            0.0,
            pending.reserved_quote - buy_cost,
        )
        currency_balances[(pending.buy_ex, pending.asset)] += size
        currency_balances[(pending.sell_ex, pending.asset)] += max(
            0.0,
            pending.reserved_asset - size,
        )
        currency_balances[(pending.sell_ex, "USDT")] += sell_proceeds

    return L2Trade(
        ts=pending.ts,
        exec_ts=pending.exec_ts,
        asset=pending.asset,
        buy_ex=pending.buy_ex,
        sell_ex=pending.sell_ex,
        size=size,
        buy_cost=buy_cost,
        sell_proceeds=sell_proceeds,
        pnl=pnl,
        signal_edge_bps=pending.signal_edge_bps,
        signal_expected_pnl=pending.signal_expected_pnl,
    )


def _walk_buy_by_quote(row: dict, quote_amount: float, depth: int) -> Tuple[float, float]:
    """Spend quote into asks. Returns (gross_base_received_before_fee, quote_spent)."""
    remaining, base_received = quote_amount, 0.0
    for i in range(depth):
        p = row.get(f"asks[{i}].price")
        a = row.get(f"asks[{i}].amount")
        if p is None or a is None or not np.isfinite(p) or not np.isfinite(a) or p <= 0 or a <= 0:
            continue
        spend = min(remaining, a * p)
        base_received += spend / p
        remaining -= spend
        if remaining <= 1e-12:
            break
    return base_received, quote_amount - remaining


def _pair_defs_from_panels(
    pair_panels: Dict[Tuple[str, str], pd.DataFrame],
) -> Dict[str, Tuple[str, str]]:
    defs: Dict[str, Tuple[str, str]] = {}
    for (_, pair), df in pair_panels.items():
        if df.empty or "base" not in df.columns or "quote" not in df.columns:
            continue
        base = str(df["base"].iloc[0])
        quote = str(df["quote"].iloc[0])
        existing = defs.get(pair)
        if existing is not None and existing != (base, quote):
            raise ValueError(f"Conflicting pair definition for {pair}: {existing} vs {(base, quote)}")
        defs[pair] = (base, quote)
    return defs


def _align_exchange_pairs(
    pair_panels: Dict[Tuple[str, str], pd.DataFrame],
    exchange: str,
) -> _ExchangePairView | None:
    pairs = sorted(pair for (ex, pair) in pair_panels if ex == exchange)
    if len(pairs) < 3:
        return None
    tss = sorted(set().union(*[set(pair_panels[(exchange, pair)]["ts"]) for pair in pairs]))
    if not tss:
        return None
    idx = pd.Index(tss, name="ts")
    aligned: Dict[str, dict] = {}
    for pair in pairs:
        df = pair_panels[(exchange, pair)].copy()
        df["source_ts"] = df["ts"]
        df = df.set_index("ts").reindex(idx).ffill()
        aligned[pair] = df.to_dict("index")
    return _ExchangePairView(
        exchange=exchange,
        pairs=pairs,
        tss=tss,
        tss_set=set(tss),
        aligned=aligned,
    )


def _build_l2_exchange_graph(
    rows_by_pair: Dict[str, dict],
    pair_defs: Dict[str, Tuple[str, str]],
    config: L2Config,
    t: int,
) -> Tuple[ExchangeGraph, Dict[Tuple[str, str], _PairAction]]:
    graph = ExchangeGraph(fee=config.fee)
    actions: Dict[Tuple[str, str], _PairAction] = {}
    for pair, row in rows_by_pair.items():
        if row is None or not _is_fresh(row, t, config.max_quote_age_ms):
            continue
        base, quote = pair_defs[pair]
        ask = row.get("asks[0].price")
        bid = row.get("bids[0].price")
        if ask is not None and np.isfinite(ask) and ask > 0:
            graph.add_directed(quote, base, 1.0 / ask)
            actions[(quote, base)] = _PairAction(pair=pair, side="buy")
        if bid is not None and np.isfinite(bid) and bid > 0:
            graph.add_directed(base, quote, bid)
            actions[(base, quote)] = _PairAction(pair=pair, side="sell")
    return graph, actions


def _rotate_cycle_to(cycle: ArbitrageCycle, start_currency: str) -> List[str]:
    nodes = cycle.currencies[:-1]
    i = nodes.index(start_currency)
    return nodes[i:] + nodes[:i] + [start_currency]


def _best_triangle_from(
    graph: ExchangeGraph,
    start_currency: str,
) -> Tuple[ArbitrageCycle, List[str]] | None:
    best: Tuple[ArbitrageCycle, List[str]] | None = None
    for cycle in enumerate_triangles(graph):
        if cycle.length != 3 or start_currency not in cycle.currencies[:-1]:
            continue
        rotated = _rotate_cycle_to(cycle, start_currency)
        if best is None or cycle.total_weight < best[0].total_weight:
            best = (cycle, rotated)
    return best


def _currency_value_usdt(
    currency: str,
    rows_by_pair: Dict[str, dict],
    pair_defs: Dict[str, Tuple[str, str]],
) -> float | None:
    if currency == "USDT":
        return 1.0
    for pair, (base, quote) in pair_defs.items():
        row = rows_by_pair.get(pair)
        if row is None:
            continue
        ask = row.get("asks[0].price")
        bid = row.get("bids[0].price")
        if ask is None or bid is None or not np.isfinite(ask) or not np.isfinite(bid):
            continue
        if ask <= 0 or bid <= 0:
            continue
        mid = (float(ask) + float(bid)) / 2.0
        if base == currency and quote == "USDT":
            return mid
        if base == "USDT" and quote == currency:
            return 1.0 / mid
    return None


def _initial_currency_value_usdt(
    view: _ExchangePairView,
    currency: str,
    pair_defs: Dict[str, Tuple[str, str]],
    max_age_ms: int,
) -> float | None:
    for t in view.tss:
        rows_by_pair = _rows_at(view, t)
        fresh_rows = {
            pair: row
            for pair, row in rows_by_pair.items()
            if row is not None and _is_fresh(row, t, max_age_ms)
        }
        value = _currency_value_usdt(currency, fresh_rows, pair_defs)
        if value is not None and value > 0:
            return value
    return None


def _initial_combined_balances(
    asset_views: Dict[str, _AssetView],
    exchange_views: Dict[str, _ExchangePairView],
    pair_defs: Dict[str, Tuple[str, str]],
    config: L2Config,
) -> tuple[Dict[Tuple[str, str], float], float]:
    venues = sorted(
        {venue for view in asset_views.values() for venue in view.venues}
        | set(exchange_views)
    )
    currencies = sorted(
        {"USDT"}
        | set(asset_views)
        | {currency for pair in pair_defs.values() for currency in pair}
    )
    balances = {(venue, currency): 0.0 for venue in venues for currency in currencies}
    if not venues or not currencies:
        return balances, 0.0

    if config.inventory_per_currency_usdt is not None:
        per_currency = config.inventory_per_currency_usdt
        for venue in venues:
            for currency in currencies:
                if currency == "USDT":
                    balances[(venue, currency)] = per_currency
                    continue
                value = None
                view = exchange_views.get(venue)
                if view is not None:
                    value = _initial_currency_value_usdt(
                        view,
                        currency,
                        pair_defs,
                        config.max_quote_age_ms,
                    )
                if value is None:
                    asset_view = asset_views.get(currency)
                    if asset_view is not None and venue in asset_view.venues:
                        value = _initial_mid(
                            asset_view.aligned,
                            asset_view.tss,
                            venue,
                            config.max_quote_age_ms,
                        )
                balances[(venue, currency)] = (
                    per_currency / value
                    if value is not None and value > 0
                    else 0.0
                )
        return balances, per_currency * len(venues) * len(currencies)

    quote_per_venue = config.start_capital_usdt / len(venues)
    for venue in venues:
        balances[(venue, "USDT")] = quote_per_venue
    return balances, config.start_capital_usdt


def _execute_triangular_cycle(
    cycle: List[str],
    actions: Dict[Tuple[str, str], _PairAction],
    rows_by_pair: Dict[str, dict],
    amount: float,
    config: L2Config,
) -> float | None:
    fee_factor = 1.0 - config.fee
    if fee_factor <= 0 or amount <= 0:
        return None

    current_amount = amount
    for u, v in zip(cycle, cycle[1:]):
        action = actions.get((u, v))
        if action is None:
            return None
        row = rows_by_pair.get(action.pair)
        if row is None:
            return None

        if action.side == "buy":
            gross_received, spent = _walk_buy_by_quote(row, current_amount, config.depth)
            if spent < current_amount * (1.0 - 1e-9) or gross_received <= 0:
                return None
            current_amount = gross_received * fee_factor
        else:
            filled, proceeds = _walk_sell(row, current_amount, config.depth)
            if filled < current_amount * (1.0 - 1e-9) or proceeds <= 0:
                return None
            current_amount = proceeds * fee_factor

    return current_amount if cycle[0] == cycle[-1] else None


def _rows_at(view: _ExchangePairView, t: int) -> Dict[str, dict]:
    return {pair: view.aligned[pair].get(t) for pair in view.pairs}


def _execute_pending_triangle(
    pending: _PendingTriangularTrade,
    view: _ExchangePairView,
    pair_defs: Dict[str, Tuple[str, str]],
    config: L2Config,
) -> L2TriangularTrade | None:
    rows_by_pair = _rows_at(view, pending.exec_ts)
    _, actions = _build_l2_exchange_graph(rows_by_pair, pair_defs, config, pending.exec_ts)
    value_usdt = _currency_value_usdt(pending.reserved_currency, rows_by_pair, pair_defs)
    if value_usdt is None or value_usdt <= 0:
        return None
    end_amount = _execute_triangular_cycle(
        pending.cycle,
        actions,
        rows_by_pair,
        pending.reserved_amount,
        config,
    )
    if end_amount is None:
        return None
    start_value_usdt = pending.reserved_amount * value_usdt
    end_value_usdt = end_amount * value_usdt
    return L2TriangularTrade(
        ts=pending.ts,
        exec_ts=pending.exec_ts,
        exchange=pending.exchange,
        cycle=pending.cycle,
        start_currency=pending.reserved_currency,
        start_amount=pending.reserved_amount,
        end_amount=end_amount,
        start_value_usdt=start_value_usdt,
        end_value_usdt=end_value_usdt,
        pnl=end_value_usdt - start_value_usdt,
        signal_edge_bps=pending.signal_edge_bps,
        signal_expected_pnl=pending.signal_expected_pnl,
    )


def run_l2_backtest(
    panels: Dict[Tuple[str, str], pd.DataFrame],
    config: L2Config,
) -> L2Result:
    assets = sorted({a for (_, a) in panels})
    asset_views: Dict[str, _AssetView] = {}
    for asset in assets:
        venues, tss, aligned = _align_asset(panels, asset)
        if len(venues) >= 2 and tss:
            asset_views[asset] = _AssetView(
                venues=venues,
                tss=tss,
                tss_set=set(tss),
                aligned=aligned,
            )

    trades: List[L2Trade] = []
    raw_crosses = 0
    executable_candidates = 0
    inventory_skips = 0
    pending: list[tuple[int, int, _PendingTrade]] = []
    seq = 0
    quote_balances, asset_balances, start_capital = _initial_balances(asset_views, config)
    pnl_by_exec_ts: Dict[int, float] = {}

    all_ts = sorted(set().union(*[set(df["ts"]) for df in panels.values()]))
    for t in all_ts:
        while pending and pending[0][0] <= t:
            _, _, pending_trade = heappop(pending)
            view = asset_views[pending_trade.asset]
            trade = _execute_pending_trade(
                pending_trade,
                view,
                config,
                quote_balances,
                asset_balances,
            )
            if trade is None:
                if config.enforce_inventory:
                    _release_pending_reservation(
                        pending_trade,
                        quote_balances,
                        asset_balances,
                    )
                continue
            executable_candidates += 1
            trades.append(trade)
            pnl_by_exec_ts[trade.exec_ts] = pnl_by_exec_ts.get(trade.exec_ts, 0.0) + trade.pnl

        for asset, view in asset_views.items():
            if t not in view.tss_set:
                continue
            n = bisect_left(view.tss, t)
            te_idx = bisect_left(view.tss, t + config.latency_ms, lo=n)
            if te_idx >= len(view.tss):
                continue
            te = view.tss[te_idx]

            best_ask_ex, best_ask = None, float("inf")
            best_bid_ex, best_bid = None, float("-inf")
            for ex in view.venues:
                row = view.aligned[ex].get(t)
                if row is None or not _is_fresh(row, t, config.max_quote_age_ms):
                    continue
                ask = row.get("asks[0].price")
                bid = row.get("bids[0].price")
                if ask is not None and np.isfinite(ask) and ask < best_ask:
                    best_ask, best_ask_ex = ask, ex
                if bid is not None and np.isfinite(bid) and bid > best_bid:
                    best_bid, best_bid_ex = bid, ex

            if best_ask_ex is None or best_bid_ex is None or best_ask_ex == best_bid_ex:
                continue
            if best_bid <= best_ask:
                continue
            raw_crosses += 1

            fee_factor = 1.0 - config.fee
            if fee_factor <= 0:
                continue
            if config.enforce_inventory:
                if config.stake_fraction is None:
                    reserved_quote = _fixed_or_fractional_notional(config)
                else:
                    stake_fraction = _stake_fraction(config)
                    quote_capacity = quote_balances.get(best_ask_ex, 0.0) * stake_fraction
                    asset_capacity = (
                        asset_balances.get((best_bid_ex, asset), 0.0)
                        * stake_fraction
                        * best_ask
                        / fee_factor
                    )
                    reserved_quote = min(quote_capacity, asset_capacity)
            else:
                reserved_quote = _fixed_or_fractional_notional(config)
            _, reserved_quote = _apply_notional_cap(1.0, reserved_quote, config)
            if reserved_quote <= 0:
                continue

            signal_gross_edge = (best_bid / best_ask) * fee_factor**2 - 1.0
            signal_edge_bps = signal_gross_edge * 10_000
            signal_expected_pnl = reserved_quote * signal_gross_edge
            if (
                signal_expected_pnl < _min_signal_profit_usdt(config, reserved_quote)
                or signal_edge_bps < _min_signal_edge_bps(config)
            ):
                continue

            reserved_asset = (reserved_quote / best_ask) * fee_factor
            if config.enforce_inventory:
                if quote_balances.get(best_ask_ex, 0.0) < reserved_quote:
                    inventory_skips += 1
                    continue
                if asset_balances.get((best_bid_ex, asset), 0.0) < reserved_asset:
                    inventory_skips += 1
                    continue
                quote_balances[best_ask_ex] -= reserved_quote
                asset_balances[(best_bid_ex, asset)] -= reserved_asset

            pending_trade = _PendingTrade(
                seq=seq,
                ts=t,
                exec_ts=te,
                asset=asset,
                buy_ex=best_ask_ex,
                sell_ex=best_bid_ex,
                reserved_quote=reserved_quote,
                reserved_asset=reserved_asset,
                signal_edge_bps=signal_edge_bps,
                signal_expected_pnl=signal_expected_pnl,
            )
            heappush(pending, (te, seq, pending_trade))
            seq += 1

    while pending:
        _, _, pending_trade = heappop(pending)
        view = asset_views[pending_trade.asset]
        trade = _execute_pending_trade(
            pending_trade,
            view,
            config,
            quote_balances,
            asset_balances,
        )
        if trade is None:
            if config.enforce_inventory:
                _release_pending_reservation(
                    pending_trade,
                    quote_balances,
                    asset_balances,
                )
            continue
        executable_candidates += 1
        trades.append(trade)
        pnl_by_exec_ts[trade.exec_ts] = pnl_by_exec_ts.get(trade.exec_ts, 0.0) + trade.pnl

    # Equity curve built from CHRONOLOGICALLY SORTED trades across all assets,
    # so the curve is time-consistent regardless of per-asset processing order.
    pnl_at = pd.Series(0.0, index=all_ts)
    for ts, pnl in pnl_by_exec_ts.items():
        if ts in pnl_at.index:
            pnl_at.loc[ts] += pnl
    equity = start_capital + pnl_at.cumsum()
    equity.index = pd.to_datetime(all_ts, unit="ms", utc=True)

    return L2Result(
        trades=sorted(trades, key=lambda x: (x.exec_ts, x.ts)),
        equity_curve=equity,
        grid_points=len(all_ts),
        raw_crosses=raw_crosses,
        executable_candidates=executable_candidates,
        inventory_skips=inventory_skips,
        start_capital_usdt=start_capital,
    )


def run_l2_triangular_backtest(
    pair_panels: Dict[Tuple[str, str], pd.DataFrame],
    config: L2Config,
    start_currency: str = "USDT",
) -> L2TriangularResult:
    """Same-exchange triangular L2 arbitrage with signal-time graph search.

    The signal graph is built only from books visible at t. If a signal passes
    filters, execution is attempted on the first grid point at/after
    t + latency_ms using those execution-time books.
    """
    pair_defs = _pair_defs_from_panels(pair_panels)
    exchange_views: Dict[str, _ExchangePairView] = {}
    for exchange in sorted({ex for ex, _ in pair_panels}):
        view = _align_exchange_pairs(pair_panels, exchange)
        if view is not None:
            exchange_views[exchange] = view

    if not exchange_views:
        return L2TriangularResult(start_capital_usdt=config.start_capital_usdt)

    all_ts = sorted(set().union(*[set(df["ts"]) for df in pair_panels.values()]))
    currencies = sorted({currency for pair in pair_defs.values() for currency in pair})
    if config.inventory_per_currency_usdt is not None:
        start_capital = config.inventory_per_currency_usdt * len(exchange_views) * len(currencies)
    else:
        start_capital = config.start_capital_usdt
    currency_balances: Dict[Tuple[str, str], float] = {}
    for exchange, view in exchange_views.items():
        for currency in currencies:
            if config.inventory_per_currency_usdt is not None:
                value = _initial_currency_value_usdt(
                    view,
                    currency,
                    pair_defs,
                    config.max_quote_age_ms,
                )
                currency_balances[(exchange, currency)] = (
                    config.inventory_per_currency_usdt / value
                    if value is not None and value > 0
                    else 0.0
                )
            elif currency == start_currency:
                currency_balances[(exchange, currency)] = config.start_capital_usdt / len(exchange_views)
            else:
                currency_balances[(exchange, currency)] = 0.0
    pending: list[tuple[int, int, _PendingTriangularTrade]] = []
    trades: List[L2TriangularTrade] = []
    pnl_by_exec_ts: Dict[int, float] = {}
    raw_cycles = 0
    executable_candidates = 0
    inventory_skips = 0
    seq = 0

    def handle_pending(pending_trade: _PendingTriangularTrade) -> None:
        nonlocal executable_candidates
        view = exchange_views[pending_trade.exchange]
        trade = _execute_pending_triangle(pending_trade, view, pair_defs, config)
        if trade is None:
            if config.enforce_inventory:
                currency_balances[
                    (pending_trade.exchange, pending_trade.reserved_currency)
                ] += pending_trade.reserved_amount
            return
        executable_candidates += 1
        trades.append(trade)
        pnl_by_exec_ts[trade.exec_ts] = pnl_by_exec_ts.get(trade.exec_ts, 0.0) + trade.pnl
        if config.enforce_inventory:
            currency_balances[(trade.exchange, trade.start_currency)] += trade.end_amount

    for t in all_ts:
        while pending and pending[0][0] <= t:
            _, _, pending_trade = heappop(pending)
            handle_pending(pending_trade)

        for exchange, view in exchange_views.items():
            if t not in view.tss_set:
                continue
            n = bisect_left(view.tss, t)
            te_idx = bisect_left(view.tss, t + config.latency_ms, lo=n)
            if te_idx >= len(view.tss):
                continue
            te = view.tss[te_idx]

            rows_by_pair = _rows_at(view, t)
            graph, actions = _build_l2_exchange_graph(rows_by_pair, pair_defs, config, t)
            best_signal = None
            seen_cycles: set[tuple[str, ...]] = set()
            for signal_cycle in enumerate_triangles(graph):
                if signal_cycle.length != 3:
                    continue
                cycle_key = _canonical_cycle_key(signal_cycle)
                if cycle_key in seen_cycles:
                    continue
                seen_cycles.add(cycle_key)
                raw_cycles += 1
                for cycle_start in signal_cycle.currencies[:-1]:
                    cycle_path = _rotate_cycle_to(signal_cycle, cycle_start)
                    value_usdt = _currency_value_usdt(cycle_start, rows_by_pair, pair_defs)
                    if value_usdt is None or value_usdt <= 0:
                        continue
                    balance = currency_balances.get((exchange, cycle_start), 0.0)
                    stake_amount = balance * _stake_fraction(config)
                    stake_value_usdt = stake_amount * value_usdt
                    stake_amount, stake_value_usdt = _apply_notional_cap(
                        stake_amount,
                        stake_value_usdt,
                        config,
                    )
                    if stake_amount <= 0 or stake_value_usdt <= 0:
                        continue
                    expected_end = _execute_triangular_cycle(
                        cycle_path,
                        actions,
                        rows_by_pair,
                        stake_amount,
                        config,
                    )
                    if expected_end is None:
                        continue
                    signal_gross_edge = expected_end / stake_amount - 1.0
                    signal_edge_bps = signal_gross_edge * 10_000
                    signal_expected_pnl = stake_value_usdt * signal_gross_edge
                    if (
                        signal_expected_pnl < _min_signal_profit_usdt(config, stake_value_usdt)
                        or signal_edge_bps < _min_signal_edge_bps(config)
                    ):
                        continue
                    if best_signal is None or signal_expected_pnl > best_signal["expected_pnl"]:
                        best_signal = {
                            "cycle_path": cycle_path,
                            "currency": cycle_start,
                            "amount": stake_amount,
                            "value_usdt": stake_value_usdt,
                            "edge_bps": signal_edge_bps,
                            "expected_pnl": signal_expected_pnl,
                        }
            if best_signal is None:
                continue

            if config.enforce_inventory:
                balance_key = (exchange, best_signal["currency"])
                if currency_balances.get(balance_key, 0.0) < best_signal["amount"]:
                    inventory_skips += 1
                    continue
                currency_balances[balance_key] -= best_signal["amount"]

            pending_trade = _PendingTriangularTrade(
                seq=seq,
                ts=t,
                exec_ts=te,
                exchange=exchange,
                cycle=best_signal["cycle_path"],
                reserved_currency=best_signal["currency"],
                reserved_amount=best_signal["amount"],
                signal_start_value_usdt=best_signal["value_usdt"],
                signal_edge_bps=best_signal["edge_bps"],
                signal_expected_pnl=best_signal["expected_pnl"],
            )
            heappush(pending, (te, seq, pending_trade))
            seq += 1

    while pending:
        _, _, pending_trade = heappop(pending)
        handle_pending(pending_trade)

    pnl_at = pd.Series(0.0, index=all_ts)
    for ts, pnl in pnl_by_exec_ts.items():
        if ts in pnl_at.index:
            pnl_at.loc[ts] += pnl
    equity = start_capital + pnl_at.cumsum()
    equity.index = pd.to_datetime(all_ts, unit="ms", utc=True)

    return L2TriangularResult(
        trades=sorted(trades, key=lambda x: (x.exec_ts, x.ts)),
        equity_curve=equity,
        grid_points=len(all_ts),
        raw_cycles=raw_cycles,
        executable_candidates=executable_candidates,
        inventory_skips=inventory_skips,
        start_capital_usdt=start_capital,
    )


def run_l2_combined_backtest(
    panels: Dict[Tuple[str, str], pd.DataFrame],
    pair_panels: Dict[Tuple[str, str], pd.DataFrame],
    config: L2Config,
) -> L2CombinedResult:
    """Unified direct + same-exchange triangle L2 strategy.

    At each signal timestamp, direct cross-exchange candidates and length-3
    same-exchange triangle candidates compete for one shared pre-funded
    (exchange, currency) inventory. The best expected-PnL signal is reserved
    and later executed on the latency-shifted book.
    """
    assets = sorted({a for (_, a) in panels})
    asset_views: Dict[str, _AssetView] = {}
    for asset in assets:
        venues, tss, aligned = _align_asset(panels, asset)
        if len(venues) >= 2 and tss:
            asset_views[asset] = _AssetView(
                venues=venues,
                tss=tss,
                tss_set=set(tss),
                aligned=aligned,
            )

    pair_defs = _pair_defs_from_panels(pair_panels)
    exchange_views: Dict[str, _ExchangePairView] = {}
    for exchange in sorted({ex for ex, _ in pair_panels}):
        view = _align_exchange_pairs(pair_panels, exchange)
        if view is not None:
            exchange_views[exchange] = view

    all_frames = list(panels.values()) + list(pair_panels.values())
    if not all_frames:
        return L2CombinedResult(start_capital_usdt=config.start_capital_usdt)
    all_ts = sorted(set().union(*[set(df["ts"]) for df in all_frames]))

    currency_balances, start_capital = _initial_combined_balances(
        asset_views,
        exchange_views,
        pair_defs,
        config,
    )
    pending: list[tuple[int, int, str, _PendingTrade | _PendingTriangularTrade]] = []
    direct_trades: List[L2Trade] = []
    triangular_trades: List[L2TriangularTrade] = []
    pnl_by_exec_ts: Dict[int, float] = {}
    raw_crosses = 0
    raw_cycles = 0
    executable_candidates = 0
    inventory_skips = 0
    seq = 0

    def handle_pending(kind: str, pending_trade: _PendingTrade | _PendingTriangularTrade) -> None:
        nonlocal executable_candidates
        if kind == "direct":
            assert isinstance(pending_trade, _PendingTrade)
            view = asset_views[pending_trade.asset]
            trade = _execute_pending_direct_currency(
                pending_trade,
                view,
                config,
                currency_balances,
            )
            if trade is None:
                if config.enforce_inventory:
                    _release_direct_currency_reservation(pending_trade, currency_balances)
                return
            executable_candidates += 1
            direct_trades.append(trade)
            pnl_by_exec_ts[trade.exec_ts] = pnl_by_exec_ts.get(trade.exec_ts, 0.0) + trade.pnl
            return

        assert isinstance(pending_trade, _PendingTriangularTrade)
        view = exchange_views[pending_trade.exchange]
        trade = _execute_pending_triangle(pending_trade, view, pair_defs, config)
        if trade is None:
            if config.enforce_inventory:
                currency_balances[
                    (pending_trade.exchange, pending_trade.reserved_currency)
                ] += pending_trade.reserved_amount
            return
        executable_candidates += 1
        triangular_trades.append(trade)
        pnl_by_exec_ts[trade.exec_ts] = pnl_by_exec_ts.get(trade.exec_ts, 0.0) + trade.pnl
        if config.enforce_inventory:
            currency_balances[(trade.exchange, trade.start_currency)] += trade.end_amount

    def maybe_update_best(best: _CombinedSignal | None, signal: _CombinedSignal) -> _CombinedSignal:
        if best is None or signal.expected_pnl > best.expected_pnl:
            return signal
        return best

    for t in all_ts:
        while pending and pending[0][0] <= t:
            _, _, kind, pending_trade = heappop(pending)
            handle_pending(kind, pending_trade)

        best_signal: _CombinedSignal | None = None

        for asset, view in asset_views.items():
            if t not in view.tss_set:
                continue
            n = bisect_left(view.tss, t)
            te_idx = bisect_left(view.tss, t + config.latency_ms, lo=n)
            if te_idx >= len(view.tss):
                continue
            te = view.tss[te_idx]

            best_ask_ex, best_ask = None, float("inf")
            best_bid_ex, best_bid = None, float("-inf")
            for ex in view.venues:
                row = view.aligned[ex].get(t)
                if row is None or not _is_fresh(row, t, config.max_quote_age_ms):
                    continue
                ask = row.get("asks[0].price")
                bid = row.get("bids[0].price")
                if ask is not None and np.isfinite(ask) and ask < best_ask:
                    best_ask, best_ask_ex = ask, ex
                if bid is not None and np.isfinite(bid) and bid > best_bid:
                    best_bid, best_bid_ex = bid, ex

            if best_ask_ex is None or best_bid_ex is None or best_ask_ex == best_bid_ex:
                continue
            if best_bid <= best_ask:
                continue
            raw_crosses += 1

            fee_factor = 1.0 - config.fee
            if fee_factor <= 0:
                continue
            if config.enforce_inventory:
                if config.stake_fraction is None:
                    reserved_quote = _fixed_or_fractional_notional(config)
                else:
                    stake_fraction = _stake_fraction(config)
                    quote_capacity = (
                        currency_balances.get((best_ask_ex, "USDT"), 0.0)
                        * stake_fraction
                    )
                    asset_capacity = (
                        currency_balances.get((best_bid_ex, asset), 0.0)
                        * stake_fraction
                        * best_ask
                        / fee_factor
                    )
                    reserved_quote = min(quote_capacity, asset_capacity)
            else:
                reserved_quote = _fixed_or_fractional_notional(config)
            _, reserved_quote = _apply_notional_cap(1.0, reserved_quote, config)
            if reserved_quote <= 0:
                continue

            signal_gross_edge = (best_bid / best_ask) * fee_factor**2 - 1.0
            signal_edge_bps = signal_gross_edge * 10_000
            signal_expected_pnl = reserved_quote * signal_gross_edge
            if (
                signal_expected_pnl < _min_signal_profit_usdt(config, reserved_quote)
                or signal_edge_bps < _min_signal_edge_bps(config)
            ):
                continue

            reserved_asset = (reserved_quote / best_ask) * fee_factor
            signal = _CombinedSignal(
                kind="direct",
                exec_ts=te,
                expected_pnl=signal_expected_pnl,
                edge_bps=signal_edge_bps,
                asset=asset,
                buy_ex=best_ask_ex,
                sell_ex=best_bid_ex,
                reserved_quote=reserved_quote,
                reserved_asset=reserved_asset,
            )
            best_signal = maybe_update_best(best_signal, signal)

        for exchange, view in exchange_views.items():
            if t not in view.tss_set:
                continue
            n = bisect_left(view.tss, t)
            te_idx = bisect_left(view.tss, t + config.latency_ms, lo=n)
            if te_idx >= len(view.tss):
                continue
            te = view.tss[te_idx]

            rows_by_pair = _rows_at(view, t)
            graph, actions = _build_l2_exchange_graph(rows_by_pair, pair_defs, config, t)
            seen_cycles: set[tuple[str, ...]] = set()
            for signal_cycle in enumerate_triangles(graph):
                if signal_cycle.length != 3:
                    continue
                cycle_key = _canonical_cycle_key(signal_cycle)
                if cycle_key in seen_cycles:
                    continue
                seen_cycles.add(cycle_key)
                raw_cycles += 1
                for cycle_start in signal_cycle.currencies[:-1]:
                    cycle_path = _rotate_cycle_to(signal_cycle, cycle_start)
                    value_usdt = _currency_value_usdt(cycle_start, rows_by_pair, pair_defs)
                    if value_usdt is None or value_usdt <= 0:
                        continue
                    balance = currency_balances.get((exchange, cycle_start), 0.0)
                    stake_amount = balance * _stake_fraction(config)
                    stake_value_usdt = stake_amount * value_usdt
                    stake_amount, stake_value_usdt = _apply_notional_cap(
                        stake_amount,
                        stake_value_usdt,
                        config,
                    )
                    if stake_amount <= 0 or stake_value_usdt <= 0:
                        continue
                    expected_end = _execute_triangular_cycle(
                        cycle_path,
                        actions,
                        rows_by_pair,
                        stake_amount,
                        config,
                    )
                    if expected_end is None:
                        continue
                    signal_gross_edge = expected_end / stake_amount - 1.0
                    signal_edge_bps = signal_gross_edge * 10_000
                    signal_expected_pnl = stake_value_usdt * signal_gross_edge
                    if (
                        signal_expected_pnl < _min_signal_profit_usdt(config, stake_value_usdt)
                        or signal_edge_bps < _min_signal_edge_bps(config)
                    ):
                        continue
                    signal = _CombinedSignal(
                        kind="triangle",
                        exec_ts=te,
                        expected_pnl=signal_expected_pnl,
                        edge_bps=signal_edge_bps,
                        exchange=exchange,
                        cycle=cycle_path,
                        reserved_currency=cycle_start,
                        reserved_amount=stake_amount,
                        start_value_usdt=stake_value_usdt,
                    )
                    best_signal = maybe_update_best(best_signal, signal)

        if best_signal is None:
            continue

        if best_signal.kind == "direct":
            if (
                best_signal.asset is None
                or best_signal.buy_ex is None
                or best_signal.sell_ex is None
            ):
                continue
            if config.enforce_inventory:
                quote_key = (best_signal.buy_ex, "USDT")
                asset_key = (best_signal.sell_ex, best_signal.asset)
                if currency_balances.get(quote_key, 0.0) < best_signal.reserved_quote:
                    inventory_skips += 1
                    continue
                if currency_balances.get(asset_key, 0.0) < best_signal.reserved_asset:
                    inventory_skips += 1
                    continue
                currency_balances[quote_key] -= best_signal.reserved_quote
                currency_balances[asset_key] -= best_signal.reserved_asset
            pending_trade = _PendingTrade(
                seq=seq,
                ts=t,
                exec_ts=best_signal.exec_ts,
                asset=best_signal.asset,
                buy_ex=best_signal.buy_ex,
                sell_ex=best_signal.sell_ex,
                reserved_quote=best_signal.reserved_quote,
                reserved_asset=best_signal.reserved_asset,
                signal_edge_bps=best_signal.edge_bps,
                signal_expected_pnl=best_signal.expected_pnl,
            )
            heappush(pending, (best_signal.exec_ts, seq, "direct", pending_trade))
            seq += 1
            continue

        if (
            best_signal.exchange is None
            or best_signal.cycle is None
            or best_signal.reserved_currency is None
        ):
            continue
        if config.enforce_inventory:
            balance_key = (best_signal.exchange, best_signal.reserved_currency)
            if currency_balances.get(balance_key, 0.0) < best_signal.reserved_amount:
                inventory_skips += 1
                continue
            currency_balances[balance_key] -= best_signal.reserved_amount
        pending_trade = _PendingTriangularTrade(
            seq=seq,
            ts=t,
            exec_ts=best_signal.exec_ts,
            exchange=best_signal.exchange,
            cycle=best_signal.cycle,
            reserved_currency=best_signal.reserved_currency,
            reserved_amount=best_signal.reserved_amount,
            signal_start_value_usdt=best_signal.start_value_usdt,
            signal_edge_bps=best_signal.edge_bps,
            signal_expected_pnl=best_signal.expected_pnl,
        )
        heappush(pending, (best_signal.exec_ts, seq, "triangle", pending_trade))
        seq += 1

    while pending:
        _, _, kind, pending_trade = heappop(pending)
        handle_pending(kind, pending_trade)

    pnl_at = pd.Series(0.0, index=all_ts)
    for ts, pnl in pnl_by_exec_ts.items():
        if ts in pnl_at.index:
            pnl_at.loc[ts] += pnl
    equity = start_capital + pnl_at.cumsum()
    equity.index = pd.to_datetime(all_ts, unit="ms", utc=True)

    return L2CombinedResult(
        direct_trades=sorted(direct_trades, key=lambda x: (x.exec_ts, x.ts)),
        triangular_trades=sorted(triangular_trades, key=lambda x: (x.exec_ts, x.ts)),
        equity_curve=equity,
        grid_points=len(all_ts),
        raw_crosses=raw_crosses,
        raw_cycles=raw_cycles,
        executable_candidates=executable_candidates,
        inventory_skips=inventory_skips,
        start_capital_usdt=start_capital,
    )
