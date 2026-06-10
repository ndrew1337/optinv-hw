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


def bounded_bellman_ford_cycles(
    graph: ExchangeGraph,
    max_len: int = 3,
    min_len: int = 2,
) -> List[ArbitrageCycle]:
    """Length-bounded Bellman-Ford: best (most-negative) cycle of length <= K
    per source currency.

    DP over `dist[k][v]` = min weight of a walk of EXACTLY k edges from the
    source to v (with predecessor edges for reconstruction). After K rounds,
    `dist[k][source] < 0` for some k in [min_len, max_len] is a profitable
    closed walk; we keep the most negative per source and reconstruct it.

    Cost: O(V * K * E) — polynomial in K (unlike enumerate's O(V^K)), so this is
    the scalable engine when the currency universe grows. Non-simple walks
    (a repeated currency) are discarded so executed cycles stay well-formed.
    """
    nodes = graph.currencies
    adj: Dict[str, List[WeightedEdge]] = {}
    for e in graph.edges:
        adj.setdefault(e.u, []).append(e)

    fee_factor = 1.0 - graph.fee
    out: List[ArbitrageCycle] = []
    seen: Set[tuple] = set()

    for s in nodes:
        dist = [dict.fromkeys(nodes, INF) for _ in range(max_len + 1)]
        pred: List[Dict[str, WeightedEdge]] = [dict() for _ in range(max_len + 1)]
        dist[0][s] = 0.0
        for k in range(1, max_len + 1):
            dk, dk1, pk = dist[k], dist[k - 1], pred[k]
            for u in nodes:
                du = dk1[u]
                if du == INF:
                    continue
                for e in adj.get(u, ()):  # noqa: B007
                    nd = du + e.weight
                    if nd < dk[e.v] - 1e-15:
                        dk[e.v] = nd
                        pk[e.v] = e

        best_k, best_w = None, -1e-12
        for k in range(min_len, max_len + 1):
            if dist[k][s] < best_w:
                best_w, best_k = dist[k][s], k
        if best_k is None:
            continue

        # reconstruct the best_k-edge walk ending back at s
        edges: List[WeightedEdge] = []
        cur, ok = s, True
        for k in range(best_k, 0, -1):
            e = pred[k].get(cur)
            if e is None:
                ok = False
                break
            edges.append(e)
            cur = e.u
        if not ok or cur != s:
            continue
        edges.reverse()
        path_nodes = [s] + [e.v for e in edges]  # [s, ..., s]
        internal = path_nodes[:-1]
        if len(set(internal)) != len(internal):  # keep only simple cycles
            continue
        key = min(tuple(internal[i:] + internal[:i]) for i in range(len(internal)))
        if key in seen:
            continue
        seen.add(key)
        gross = 1.0
        for e in edges:
            gross *= e.rate * fee_factor
        out.append(
            ArbitrageCycle(
                currencies=path_nodes,
                edges=edges,
                total_weight=sum(e.weight for e in edges),
                gross_multiplier=gross,
            )
        )

    out.sort(key=lambda c: c.total_weight)
    return out


def enumerate_cycles(
    graph: ExchangeGraph,
    max_len: int = 3,
    min_len: int = 2,
) -> List[ArbitrageCycle]:
    """All simple NEGATIVE cycles with edge-length in [min_len, max_len].

    Generalises `enumerate_triangles` to arbitrary K via depth-bounded DFS:
    a profitable arbitrage cycle (product of rates after fee > 1) is exactly a
    negative cycle (sum of -ln weights < 0). Cycles are deduped by canonical
    rotation and returned best (most negative) first.
    """
    adj: Dict[str, List[WeightedEdge]] = {}
    for e in graph.edges:
        adj.setdefault(e.u, []).append(e)

    fee_factor = 1.0 - graph.fee
    out: List[ArbitrageCycle] = []
    seen: Set[tuple] = set()

    def _canon(nodes: List[str]) -> tuple:
        return min(tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes)))

    for start in graph.currencies:
        # DFS stack: (node, path_nodes, path_edges, total_weight, visited)
        stack = [(start, [start], [], 0.0, {start})]
        while stack:
            node, pnodes, pedges, w, visited = stack.pop()
            for e in adj.get(node, ()):  # noqa: B007
                v = e.v
                if v == start:
                    clen = len(pedges) + 1
                    if min_len <= clen <= max_len and w + e.weight < -1e-12:
                        key = _canon(pnodes)
                        if key not in seen:
                            seen.add(key)
                            edges = pedges + [e]
                            gross = 1.0
                            for ee in edges:
                                gross *= ee.rate * fee_factor
                            out.append(
                                ArbitrageCycle(
                                    currencies=pnodes + [start],
                                    edges=edges,
                                    total_weight=w + e.weight,
                                    gross_multiplier=gross,
                                )
                            )
                elif v not in visited and len(pedges) < max_len - 1:
                    stack.append(
                        (v, pnodes + [v], pedges + [e], w + e.weight, visited | {v})
                    )

    out.sort(key=lambda c: c.total_weight)
    return out
