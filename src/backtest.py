"""Backtest cyclic arbitrage on 5-minute bars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .bellman_ford import find_best_cycle, enumerate_triangles
from .data import panel_close_matrix
from .graph import ExchangeGraph


@dataclass
class BacktestConfig:
    fee: float = 0.001  # 0.1% per leg (Binance spot taker default)
    half_spread_bps: float = 5.0  # half-spread per leg from mid (ask/bid proxy)
    slippage_bps: float = 2.0  # extra execution cost per leg
    min_gross_multiplier: float = 1.002  # require ~0.2% edge before trade
    max_cycle_len: int = 4
    start_capital_usdt: float = 10_000.0
    trade_fraction: float = 0.1  # deploy at most 10% of equity per opportunity
    use_triangle_only: bool = True  # short cycles per presentation
    use_intrabar_hilo: bool = False  # optimistic; not simultaneously executable


@dataclass
class TradeRecord:
    time: pd.Timestamp
    cycle: List[str]
    gross_multiplier: float
    net_multiplier: float
    pnl_usdt: float
    capital_after: float
    signal_time: Optional[pd.Timestamp] = None
    realized_gross_multiplier: Optional[float] = None


@dataclass
class BacktestResult:
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    opportunities_found: int = 0
    bars_scanned: int = 0

    def summary(self) -> Dict:
        if self.equity_curve.empty:
            return {"bars": self.bars_scanned, "trades": 0}
        ret = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        rets = self.equity_curve.pct_change().dropna()
        sharpe = (
            (rets.mean() / rets.std() * np.sqrt(288 * 365)) if rets.std() > 0 else 0.0
        )
        return {
            "bars_scanned": self.bars_scanned,
            "opportunities_found": self.opportunities_found,
            "trades_executed": len(self.trades),
            "final_capital": float(self.equity_curve.iloc[-1]),
            "total_return_pct": float(ret * 100),
            "max_drawdown_pct": float(_max_drawdown(self.equity_curve) * 100),
            "sharpe_annualized": float(sharpe),
            "avg_trade_pnl": float(np.mean([t.pnl_usdt for t in self.trades]))
            if self.trades
            else 0.0,
        }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def apply_slippage(multiplier: float, n_legs: int, slippage_bps: float) -> float:
    slip = (1.0 - slippage_bps / 10_000) ** n_legs
    return multiplier * slip


def mid_to_bid_ask(
    prices: Dict[str, float], half_spread_bps: float
) -> tuple[Dict[str, float], Dict[str, float]]:
    h = half_spread_bps / 10_000
    ask = {s: p * (1 + h) for s, p in prices.items()}
    bid = {s: p * (1 - h) for s, p in prices.items()}
    return ask, bid


def build_graph_from_prices(
    prices: Dict[str, float],
    fee: float,
    add_implied_cross: bool = True,
    ask_prices: Optional[Dict[str, float]] = None,
    bid_prices: Optional[Dict[str, float]] = None,
) -> ExchangeGraph:
    """
    Build graph from USDT spot quotes.

    With only XXXUSDT pairs the graph is a star; implied crosses
    (BASE/QUOTE ≈ (BASE/USDT)/(QUOTE/USDT)) enable BTC–ETH–USDT cycles
  as in the presentation diagram.
    """
    g = ExchangeGraph(fee=fee)
    usdt_mid: Dict[str, float] = {}
    usdt_ask: Dict[str, float] = {}
    usdt_bid: Dict[str, float] = {}

    for symbol, mid in prices.items():
        if not symbol.endswith("USDT") or mid <= 0:
            continue
        base = symbol[:-4]
        usdt_mid[base] = mid
        usdt_ask[base] = (ask_prices or {}).get(symbol, mid)
        usdt_bid[base] = (bid_prices or {}).get(symbol, mid)
        # Sell BASE for USDT at bid; buy BASE with USDT at ask
        g.add_directed(base, "USDT", usdt_bid[base])
        g.add_directed("USDT", base, 1.0 / usdt_ask[base])

    if add_implied_cross:
        bases = list(usdt_mid.keys())
        for i, b1 in enumerate(bases):
            for b2 in bases[i + 1 :]:
                # b1 -> b2: sell b1 at its BID, buy b2 at its ASK (pay the spread).
                # rate = units of b2 per 1 b1 = bid_b1 / ask_b2.
                g.add_directed(b1, b2, usdt_bid[b1] / usdt_ask[b2])
                g.add_directed(b2, b1, usdt_bid[b2] / usdt_ask[b1])
    return g


def cycle_gross_multiplier_on_graph(
    graph: ExchangeGraph,
    currencies: List[str],
) -> Optional[float]:
    """Re-price a previously selected route on a new graph snapshot."""
    if len(currencies) < 2:
        return None

    gross = 1.0
    for u, v in zip(currencies, currencies[1:]):
        edge = next((e for e in graph._adj.get(u, []) if e.v == v), None)
        if edge is None:
            return None
        gross *= edge.rate * (1.0 - graph.fee)
    return gross


def _bid_ask_for_bar(
    panel: pd.DataFrame,
    t: pd.Timestamp,
    prices: Dict[str, float],
    config: BacktestConfig,
) -> tuple[Dict[str, float], Dict[str, float]]:
    if config.use_intrabar_hilo and {"high", "low"}.issubset(panel.columns):
        snap = panel.loc[panel["open_time"] == t]
        ask_p = dict(zip(snap["symbol"], snap["high"]))
        bid_p = dict(zip(snap["symbol"], snap["low"]))
        return ask_p, bid_p
    return mid_to_bid_ask(prices, config.half_spread_bps)


def run_backtest(panel: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    times = sorted(panel["open_time"].unique())
    capital = config.start_capital_usdt
    if not times:
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            opportunities_found=0,
            bars_scanned=0,
        )

    equity = [capital]
    eq_times = [times[0]]
    trades: List[TradeRecord] = []
    opps = 0

    for i in range(len(times) - 1):
        signal_t = times[i]
        exec_t = times[i + 1]

        signal_prices = panel_close_matrix(panel, signal_t)
        if len(signal_prices) < 3:
            equity.append(capital)
            eq_times.append(exec_t)
            continue

        signal_ask, signal_bid = _bid_ask_for_bar(
            panel,
            signal_t,
            signal_prices,
            config,
        )
        signal_graph = build_graph_from_prices(
            signal_prices,
            config.fee,
            ask_prices=signal_ask,
            bid_prices=signal_bid,
        )

        if config.use_triangle_only:
            cycles = enumerate_triangles(signal_graph)
            cycle = cycles[0] if cycles else None
        else:
            cycle = find_best_cycle(signal_graph)

        if cycle is not None and cycle.length <= config.max_cycle_len:
            if cycle.gross_multiplier >= config.min_gross_multiplier:
                opps += 1
                signal_net = apply_slippage(
                    cycle.gross_multiplier,
                    cycle.length,
                    config.slippage_bps,
                )
                if signal_net > 1.0:
                    exec_prices = panel_close_matrix(panel, exec_t)
                    exec_ask, exec_bid = _bid_ask_for_bar(
                        panel,
                        exec_t,
                        exec_prices,
                        config,
                    )
                    exec_graph = build_graph_from_prices(
                        exec_prices,
                        config.fee,
                        ask_prices=exec_ask,
                        bid_prices=exec_bid,
                    )
                    realized_gross = cycle_gross_multiplier_on_graph(
                        exec_graph,
                        cycle.currencies,
                    )
                    if realized_gross is not None:
                        net = apply_slippage(
                            realized_gross,
                            cycle.length,
                            config.slippage_bps,
                        )
                        notional = capital * config.trade_fraction
                        pnl = notional * (net - 1.0)
                        capital += pnl
                        trades.append(
                            TradeRecord(
                                time=exec_t,
                                cycle=cycle.currencies,
                                gross_multiplier=cycle.gross_multiplier,
                                net_multiplier=net,
                                pnl_usdt=pnl,
                                capital_after=capital,
                                signal_time=signal_t,
                                realized_gross_multiplier=realized_gross,
                            )
                        )

        equity.append(capital)
        eq_times.append(exec_t)

    result = BacktestResult(
        trades=trades,
        equity_curve=pd.Series(equity, index=pd.DatetimeIndex(eq_times)),
        opportunities_found=opps,
        bars_scanned=max(len(times) - 1, 0),
    )
    return result


panel_close_matrix_local = panel_close_matrix
