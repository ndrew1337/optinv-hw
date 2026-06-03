"""Directed exchange graph with log-fee edge weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

Edge = Tuple[str, str]  # (from_currency, to_currency)


@dataclass(frozen=True)
class WeightedEdge:
    u: str
    v: str
    rate: float
    weight: float


def edge_weight(rate: float, fee: float) -> float:
    """w(u,v) = -ln(r * (1-f)) as in the project slides."""
    effective = rate * (1.0 - fee)
    if effective <= 0:
        raise ValueError(f"Non-positive effective rate: {rate=}, {fee=}")
    return -math.log(effective)


def cycle_product(rates: Iterable[float], fee: float) -> float:
    """Net multiplier after fee on each leg."""
    prod = 1.0
    for r in rates:
        prod *= r * (1.0 - fee)
    return prod


class ExchangeGraph:
    """Currency graph built from QUOTE-per-BASE spot prices (e.g. BTCUSDT -> price in USDT)."""

    def __init__(self, fee: float = 0.001) -> None:
        self.fee = fee
        self.currencies: List[str] = []
        self._index: Dict[str, int] = {}
        self.edges: List[WeightedEdge] = []
        self._adj: Dict[str, List[WeightedEdge]] = {}

    def _ensure_currency(self, c: str) -> int:
        if c not in self._index:
            self._index[c] = len(self.currencies)
            self.currencies.append(c)
        return self._index[c]

    def add_directed(self, u: str, v: str, rate: float) -> None:
        """Add edge u -> v with exchange rate `rate` (units of v per 1 u)."""
        if rate <= 0:
            return
        self._ensure_currency(u)
        self._ensure_currency(v)
        w = edge_weight(rate, self.fee)
        e = WeightedEdge(u, v, rate, w)
        self.edges.append(e)
        self._adj.setdefault(u, []).append(e)

    def add_pair(self, base: str, quote: str, price: float) -> None:
        """
        Symmetric pair from mid price (both directions).
        price = how many QUOTE for 1 BASE (Binance convention).
        """
        if price <= 0:
            return
        self.add_directed(base, quote, price)
        self.add_directed(quote, base, 1.0 / price)

    def n_vertices(self) -> int:
        return len(self.currencies)

    def index_of(self, c: str) -> int:
        return self._index[c]
