"""Unit tests for graph weights and Bellman-Ford."""

import math

import pytest

from src.bellman_ford import enumerate_triangles, find_best_cycle
from src.graph import ExchangeGraph, cycle_product, edge_weight


def test_edge_weight_matches_slides():
    r, f = 2.0, 0.001
    w = edge_weight(r, f)
    assert w == pytest.approx(-math.log(r * (1 - f)))


def test_manual_triangle_arbitrage():
    """Construct rates where product > 1 after fees — BF must find cycle."""
    g = ExchangeGraph(fee=0.0)  # zero fee for clean test
    # Artificial inconsistent triangle USDT -> A -> B -> USDT
    g.add_pair("A", "USDT", 100.0)
    g.add_pair("B", "USDT", 50.0)
    # cross: 1 A = 2.2 B (mispriced)
    g.add_pair("A", "B", 2.2)

    cycle = find_best_cycle(g)
    assert cycle is not None
    assert cycle.gross_multiplier > 1.0
    assert cycle.total_weight < 0


def test_no_arbitrage_consistent_market():
    g = ExchangeGraph(fee=0.001)
    g.add_pair("BTC", "USDT", 50_000.0)
    g.add_pair("ETH", "USDT", 3_000.0)
    g.add_pair("ETH", "BTC", 3_000.0 / 50_000.0)

    tris = enumerate_triangles(g)
    assert all(t.gross_multiplier <= 1.0 + 1e-9 for t in tris)
    assert find_best_cycle(g) is None


def test_fees_kill_small_edge():
    g = ExchangeGraph(fee=0.01)  # 1% per leg
    g.add_pair("BTC", "USDT", 50_000.0)
    g.add_pair("ETH", "USDT", 3_000.0)
    g.add_pair("ETH", "BTC", 3_000.0 / 50_000.0 * 1.0001)  # tiny mispricing
    assert find_best_cycle(g) is None


def test_cycle_product():
    rates = [2.0, 0.6, 1.0]
    assert cycle_product(rates, fee=0.0) == pytest.approx(1.2)


def test_no_phantom_arbitrage_from_implied_cross():
    """A consistent market (all crosses derived from the same USDT mids) must NOT
    produce arbitrage at any spread. Guards against the inverted implied-cross
    bug, which manufactured ((1+h)/(1-h))**n profit out of nothing."""
    from src.backtest import build_graph_from_prices, mid_to_bid_ask

    prices = {"BTCUSDT": 100_000.0, "ETHUSDT": 4_000.0, "SOLUSDT": 200.0}
    for half_spread_bps in (0.0, 5.0, 20.0, 100.0):
        ask, bid = mid_to_bid_ask(prices, half_spread_bps)
        g = build_graph_from_prices(prices, fee=0.0, ask_prices=ask, bid_prices=bid)
        tris = enumerate_triangles(g)
        assert all(t.gross_multiplier <= 1.0 + 1e-9 for t in tris), (
            f"phantom arbitrage at {half_spread_bps}bps: "
            f"{[round(t.gross_multiplier, 6) for t in tris[:3]]}"
        )
