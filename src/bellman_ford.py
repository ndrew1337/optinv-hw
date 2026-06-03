"""Bellman-Ford negative cycle detection for arbitrage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .graph import ExchangeGraph, WeightedEdge

INF = float("inf")


@dataclass
class ArbitrageCycle:
    currencies: List[str]
    edges: List[WeightedEdge]
    total_weight: float
    gross_multiplier: float

    @property
    def length(self) -> int:
        return len(self.edges)


def bellman_ford_negative_cycle(
    graph: ExchangeGraph,
    source: Optional[str] = None,
) -> Optional[ArbitrageCycle]:
    """
    Find a negative cycle reachable from source (or any if source is None).
    Returns first detected cycle.
    """
    n = graph.n_vertices()
    if n == 0:
        return None

    dist: Dict[str, float] = {c: 0.0 for c in graph.currencies}
    pred: Dict[str, Optional[WeightedEdge]] = {c: None for c in graph.currencies}

    if source is not None:
        for c in graph.currencies:
            dist[c] = INF
        dist[source] = 0.0

    # Standard |V|-1 relaxations
    for _ in range(n - 1):
        updated = False
        for e in graph.edges:
            if dist[e.u] + e.weight < dist[e.v] - 1e-15:
                dist[e.v] = dist[e.u] + e.weight
                pred[e.v] = e
                updated = True
        if not updated:
            break

    # nth round: edges that still relax lie on / lead to negative cycle
    x_on_cycle: Optional[str] = None
    for e in graph.edges:
        if dist[e.u] + e.weight < dist[e.v] - 1e-15:
            x_on_cycle = e.v
            break

    if x_on_cycle is None:
        return None

    # Walk back n steps to guarantee node on cycle
    y = x_on_cycle
    for _ in range(n):
        pe = pred.get(y)
        if pe is None:
            return None
        y = pe.u

    # Trace cycle
    cycle_nodes: List[str] = [y]
    cur = y
    visited: Set[str] = set()
    while True:
        pe = pred.get(cur)
        if pe is None:
            return None
        cycle_nodes.append(pe.u)
        cur = pe.u
        if cur == y:
            break
        if cur in visited:
            break
        visited.add(cur)

    cycle_nodes.reverse()
    if cycle_nodes[0] != cycle_nodes[-1]:
        cycle_nodes.append(cycle_nodes[0])

    # Map to edges along cycle
    cycle_edges: List[WeightedEdge] = []
    for i in range(len(cycle_nodes) - 1):
        u, v = cycle_nodes[i], cycle_nodes[i + 1]
        found = None
        for e in graph._adj.get(u, []):
            if e.v == v:
                found = e
                break
        if found is None:
            return None
        cycle_edges.append(found)

    total_w = sum(e.weight for e in cycle_edges)
    gross = 1.0
    for e in cycle_edges:
        gross *= e.rate * (1.0 - graph.fee)

    return ArbitrageCycle(
        currencies=cycle_nodes,
        edges=cycle_edges,
        total_weight=total_w,
        gross_multiplier=gross,
    )


def find_best_cycle(graph: ExchangeGraph) -> Optional[ArbitrageCycle]:
    """Try each currency as source; return cycle with lowest total weight."""
    best: Optional[ArbitrageCycle] = None
    for c in graph.currencies:
        cycle = bellman_ford_negative_cycle(graph, source=c)
        if cycle is None:
            continue
        if best is None or cycle.total_weight < best.total_weight:
            best = cycle
    return best


def enumerate_triangles(graph: ExchangeGraph) -> List[ArbitrageCycle]:
    """Brute-force 3-cycles for validation / comparison."""
    idx = graph._index
    rev: Dict[int, str] = {i: c for c, i in idx.items()}
    n = len(idx)
    out: List[ArbitrageCycle] = []

    adj: Dict[int, List[WeightedEdge]] = {i: [] for i in range(n)}
    for e in graph.edges:
        adj[idx[e.u]].append(e)

    for i in range(n):
        for e_ij in adj[i]:
            j = idx[e_ij.v]
            for e_jk in adj[j]:
                k = idx[e_jk.v]
                if k == i:
                    continue
                for e_ki in adj[k]:
                    if idx[e_ki.v] != i:
                        continue
                    edges = [e_ij, e_jk, e_ki]
                    tw = sum(e.weight for e in edges)
                    if tw >= -1e-12:
                        continue
                    gross = 1.0
                    for e in edges:
                        gross *= e.rate * (1.0 - graph.fee)
                    nodes = [rev[i], rev[j], rev[k], rev[i]]
                    out.append(
                        ArbitrageCycle(
                            currencies=nodes,
                            edges=edges,
                            total_weight=tw,
                            gross_multiplier=gross,
                        )
                    )
    out.sort(key=lambda c: c.total_weight)
    return out
