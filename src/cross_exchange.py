"""Cross-exchange arbitrage graph and backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    TradeRecord,
    apply_slippage,
    cycle_gross_multiplier_on_graph,
)
from .bellman_ford import enumerate_triangles, find_best_cycle
from .graph import ExchangeGraph


def node(asset: str, exchange: str) -> str:
    return f"{asset}@{exchange}"


def _ask_bid(
    row: pd.Series, half_spread_bps: float, use_venue_hilo: bool
) -> tuple[float, float]:
    mid = float(row["close"])
    if use_venue_hilo:
        return float(row["high"]), float(row["low"])
    h = half_spread_bps / 10_000
    return mid * (1 + h), mid * (1 - h)


def build_cross_exchange_graph(
    snapshot: pd.DataFrame,
    fee: float,
    transfer_fee: float,
    half_spread_bps: float,
    use_venue_hilo: bool = False,
) -> ExchangeGraph:
    """
    Vertices: asset@exchange.
    Trade edges on one venue; transfer edges move coin between venues.
    """
    g = ExchangeGraph(fee=fee)
    usdt_ask: Dict[Tuple[str, str], float] = {}
    usdt_bid: Dict[Tuple[str, str], float] = {}

    for _, row in snapshot.iterrows():
        ex = row["exchange"]
        base = row["base"]
        ask, bid = _ask_bid(row, half_spread_bps, use_venue_hilo)
        usdt_ask[(base, ex)] = ask
        usdt_bid[(base, ex)] = bid
        g.add_directed(node(base, ex), node("USDT", ex), bid)
        g.add_directed(node("USDT", ex), node(base, ex), 1.0 / ask)

    bases = sorted({b for b, _ in usdt_bid})
    exchanges = sorted({e for _, e in usdt_bid})

    # Implied cross on same exchange
    for ex in exchanges:
        local = [b for b in bases if (b, ex) in usdt_bid]
        for i, b1 in enumerate(local):
            for b2 in local[i + 1 :]:
                # sell first asset at its BID, buy the other at its ASK (pay spread)
                g.add_directed(
                    node(b1, ex),
                    node(b2, ex),
                    usdt_bid[(b1, ex)] / usdt_ask[(b2, ex)],
                )
                g.add_directed(
                    node(b2, ex),
                    node(b1, ex),
                    usdt_bid[(b2, ex)] / usdt_ask[(b1, ex)],
                )

    # Transfer same asset between exchanges (withdrawal/deposit cost)
    xfer_rate = max(1.0 - transfer_fee, 1e-12)
    for b in bases:
        for ex1 in exchanges:
            if (b, ex1) not in usdt_bid:
                continue
            for ex2 in exchanges:
                if ex1 == ex2 or (b, ex2) not in usdt_bid:
                    continue
                g.add_directed(node(b, ex1), node(b, ex2), xfer_rate)

    return g


def scan_direct_cross_arb(
    snapshot: pd.DataFrame,
    fee: float,
    transfer_fee: float,
    half_spread_bps: float,
    use_venue_hilo: bool = False,
    lag_snapshot: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[float, str, str, str]]:
    """
    Best 2-exchange hop: buy base on cheap venue, transfer, sell on expensive venue.
    Returns (net_mult, base, buy_ex, sell_ex).
    """
    best_mult = 1.0
    best: Optional[Tuple[str, str, str]] = None
    lag = lag_snapshot if lag_snapshot is not None else snapshot
    symbols = snapshot["base"].unique()
    exchanges = snapshot["exchange"].unique()

    buy_asks: Dict[Tuple[str, str], float] = {}
    for _, row in lag.iterrows():
        base, ex = row["base"], row["exchange"]
        buy_asks[(base, ex)] = _ask_bid(row, half_spread_bps, use_venue_hilo)[0]

    sell_bids: Dict[Tuple[str, str], float] = {}
    for _, row in snapshot.iterrows():
        base, ex = row["base"], row["exchange"]
        sell_bids[(base, ex)] = _ask_bid(row, half_spread_bps, use_venue_hilo)[1]

    for base in symbols:
        for buy_ex in exchanges:
            if (base, buy_ex) not in buy_asks:
                continue
            for sell_ex in exchanges:
                if sell_ex == buy_ex or (base, sell_ex) not in sell_bids:
                    continue
                rate_buy = 1.0 / buy_asks[(base, buy_ex)]
                rate_xfer = max(1.0 - transfer_fee, 1e-12)
                rate_sell = sell_bids[(base, sell_ex)]
                gross = rate_buy * rate_xfer * rate_sell
                # Two TRADES (buy + sell) carry taker fee; the transfer is not a
                # trade — its cost is already in transfer_fee. So (1-fee)**2.
                net = gross * (1.0 - fee) ** 2
                if net > best_mult:
                    best_mult = net
                    best = (base, buy_ex, sell_ex)

    if best is None:
        return None
    return (best_mult, best[0], best[1], best[2])


def _direct_cross_multiplier_for_route(
    snapshot: pd.DataFrame,
    base: str,
    buy_ex: str,
    sell_ex: str,
    fee: float,
    transfer_fee: float,
    half_spread_bps: float,
    use_venue_hilo: bool = False,
) -> Optional[float]:
    buy_rows = snapshot.loc[
        (snapshot["base"] == base) & (snapshot["exchange"] == buy_ex)
    ]
    sell_rows = snapshot.loc[
        (snapshot["base"] == base) & (snapshot["exchange"] == sell_ex)
    ]
    if buy_rows.empty or sell_rows.empty:
        return None

    buy_ask = _ask_bid(buy_rows.iloc[0], half_spread_bps, use_venue_hilo)[0]
    sell_bid = _ask_bid(sell_rows.iloc[0], half_spread_bps, use_venue_hilo)[1]
    if buy_ask <= 0 or sell_bid <= 0:
        return None

    gross = (1.0 / buy_ask) * max(1.0 - transfer_fee, 1e-12) * sell_bid
    return gross * (1.0 - fee) ** 2


@dataclass
class CrossBacktestConfig(BacktestConfig):
    transfer_fee: float = 0.0005  # ~0.05% withdrawal friction
    use_direct_scan: bool = True  # also check simple 2-exchange hops
    use_venue_hilo: bool = False  # if True: optimistic (high/low per venue)
    lag_bars: int = 0  # buy on lagged quote (stale price), sell at current
    min_net_multiplier: float = 1.0001


def run_cross_exchange_backtest(
    panel: pd.DataFrame,
    config: CrossBacktestConfig,
) -> BacktestResult:
    panel = panel.copy()
    times = sorted(panel["open_time"].unique())
    capital = config.start_capital_usdt
    if not times:
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            opportunities_found=0,
            bars_scanned=0,
        )

    equity: List[float] = [capital]
    eq_times: List[pd.Timestamp] = [times[0]]
    trades: List[TradeRecord] = []
    opps = 0

    prev_snaps: List[pd.DataFrame] = []

    for i in range(len(times) - 1):
        signal_t = times[i]
        exec_t = times[i + 1]
        snap = panel.loc[panel["open_time"] == signal_t]

        best_mult = 1.0
        cycle_path: List[str] = []
        route_kind: Optional[str] = None
        direct_route: Optional[Tuple[str, str, str]] = None

        lag_snap = (
            prev_snaps[-config.lag_bars]
            if config.lag_bars > 0 and len(prev_snaps) >= config.lag_bars
            else None
        )

        if snap["exchange"].nunique() >= 2 and config.use_direct_scan:
            direct = scan_direct_cross_arb(
                snap,
                config.fee,
                config.transfer_fee,
                config.half_spread_bps,
                config.use_venue_hilo,
                lag_snapshot=lag_snap,
            )
            if direct and direct[0] > best_mult:
                best_mult, base, buy_ex, sell_ex = direct
                route_kind = "direct"
                direct_route = (base, buy_ex, sell_ex)
                cycle_path = [
                    f"USDT@{buy_ex}",
                    f"{base}@{buy_ex}",
                    f"{base}@{sell_ex}",
                    f"USDT@{sell_ex}",
                ]

        # The multi-hop graph path uses a single SYNCHRONOUS snapshot, i.e. it
        # assumes all venues are filled at the same instant — that is look-ahead
        # once we model a quote lag. So only use it in the zero-lag scenario;
        # with lag>0 we rely on the realistic lagged direct scan above.
        if snap["exchange"].nunique() >= 2 and config.lag_bars == 0:
            g = build_cross_exchange_graph(
                snap,
                config.fee,
                config.transfer_fee,
                config.half_spread_bps,
                config.use_venue_hilo,
            )

            if config.use_triangle_only:
                cycles = enumerate_triangles(g)
                cycle = cycles[0] if cycles else None
            else:
                cycle = find_best_cycle(g)

            if cycle is not None and cycle.gross_multiplier > best_mult:
                best_mult = cycle.gross_multiplier
                cycle_path = cycle.currencies
                route_kind = "graph"
                direct_route = None

        if best_mult >= config.min_gross_multiplier:
            opps += 1
            n_legs = max(len(cycle_path) - 1, 3)
            signal_net = apply_slippage(best_mult, n_legs, config.slippage_bps)
            if signal_net >= config.min_net_multiplier:
                exec_snap = panel.loc[panel["open_time"] == exec_t]
                realized_gross: Optional[float] = None

                if route_kind == "direct" and direct_route is not None:
                    realized_gross = _direct_cross_multiplier_for_route(
                        exec_snap,
                        direct_route[0],
                        direct_route[1],
                        direct_route[2],
                        config.fee,
                        config.transfer_fee,
                        config.half_spread_bps,
                        config.use_venue_hilo,
                    )
                elif route_kind == "graph":
                    exec_graph = build_cross_exchange_graph(
                        exec_snap,
                        config.fee,
                        config.transfer_fee,
                        config.half_spread_bps,
                        config.use_venue_hilo,
                    )
                    realized_gross = cycle_gross_multiplier_on_graph(
                        exec_graph,
                        cycle_path,
                    )

                if realized_gross is not None:
                    net = apply_slippage(realized_gross, n_legs, config.slippage_bps)
                    notional = capital * config.trade_fraction
                    pnl = notional * (net - 1.0)
                    capital += pnl
                    trades.append(
                        TradeRecord(
                            time=exec_t,
                            cycle=cycle_path,
                            gross_multiplier=best_mult,
                            net_multiplier=net,
                            pnl_usdt=pnl,
                            capital_after=capital,
                            signal_time=signal_t,
                            realized_gross_multiplier=realized_gross,
                        )
                    )

        prev_snaps.append(snap.copy())
        if len(prev_snaps) > config.lag_bars + 1:
            prev_snaps.pop(0)

        equity.append(capital)
        eq_times.append(exec_t)

    eq = pd.Series(equity, index=pd.DatetimeIndex(eq_times))

    return BacktestResult(
        trades=trades,
        equity_curve=eq,
        opportunities_found=opps,
        bars_scanned=max(len(times) - 1, 0),
    )
