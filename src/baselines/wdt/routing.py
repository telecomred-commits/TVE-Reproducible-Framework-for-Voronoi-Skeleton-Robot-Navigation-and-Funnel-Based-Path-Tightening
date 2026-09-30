"""Generación de la ruta sobre el esqueleto (artículo, sección 2.2, figura 3).

F1  η_i = S.
F2  Mientras η_i ≠ T: entre los nodos adyacentes a η_i se elige el de menor
    distancia euclidiana a T.  -> ruta inicial X_0 con n nodos.
F3  Para cada una de las n-1 conexiones de X_0 se elimina provisionalmente esa
    conexión y se repite F2 -> rutas X_1 ... X_{n-1}.
F4  Se calcula la longitud de cada ruta y se elige la más corta (X_c).

Mejora de robustez respecto a ``Reduccion.m``: F2 se implementa como una
búsqueda en profundidad voraz con retroceso.  Mientras no haya callejones sin
salida produce exactamente la misma ruta que el artículo, pero si el nodo
voraz lleva a un punto muerto retrocede en lugar de quedar en un ciclo
infinito, así que siempre encuentra ruta si existe.

También se ofrece ``dijkstra`` (ruta óptima sobre el grafo) como optimización.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from common.skeleton import SkeletonGraph


@dataclass
class Route:
    nodes: list[int]
    edges: list[int]
    length: float
    polyline: np.ndarray
    label: str = ""


def _route_from(g: SkeletonGraph, nodes: list[int], edges: list[int], label: str) -> Route:
    parts = []
    for i, eid in enumerate(edges):
        pts = g.oriented_pts(eid, nodes[i])
        parts.append(pts if i == 0 else pts[1:])
    poly = np.vstack(parts) if parts else np.asarray([g.nodes[nodes[0]]])
    length = float(sum(g.edges[e].length for e in edges))
    return Route(list(nodes), list(edges), length, poly, label)


def _best_edges(g: SkeletonGraph, u: int, banned: set[frozenset]) -> dict[int, int]:
    """Vecino -> arista más corta hacia él (multigrafo), excluyendo prohibidas."""
    best: dict[int, int] = {}
    for eid in g.adj[u]:
        v = g.other(eid, u)
        if v == u or frozenset((u, v)) in banned:
            continue
        if v not in best or g.edges[eid].length < g.edges[best[v]].length:
            best[v] = eid
    return best


def greedy_route(g: SkeletonGraph, s: int, t: int, banned: set[frozenset] | None = None,
                 max_expansions: int = 200000) -> tuple[list[int], list[int]] | None:
    """F2: avance voraz hacia el nodo más cercano a T (con retroceso)."""
    banned = banned or set()
    target = g.nodes[t]

    def ordered(u):
        best = _best_edges(g, u, banned)
        items = list(best.items())
        items.sort(key=lambda it: (float(np.hypot(*(g.nodes[it[0]] - target))), g.edges[it[1]].length))
        return iter(items)

    nodes = [s]
    edges: list[int] = []
    visited = {s}
    stack = [ordered(s)]
    expansions = 0
    while stack:
        if nodes[-1] == t:
            return nodes, edges
        expansions += 1
        if expansions > max_expansions:
            return None
        nxt = None
        for v, eid in stack[-1]:
            if v not in visited:
                nxt = (v, eid)
                break
        if nxt is None:         # callejón sin salida -> retroceder
            stack.pop()
            nodes.pop()
            if edges:
                edges.pop()
            continue
        v, eid = nxt
        visited.add(v)
        nodes.append(v)
        edges.append(eid)
        stack.append(ordered(v))
    return None


def paper_routes(g: SkeletonGraph, s: int, t: int) -> tuple[list[Route], int]:
    """F1-F4.  Devuelve todas las rutas candidatas y el índice de la elegida."""
    first = greedy_route(g, s, t)
    if first is None:
        return [], -1
    routes = [_route_from(g, *first, label="X0 (ruta inicial)")]
    seen = {tuple(first[1])}
    n0 = first[0]
    for y in range(len(n0) - 1):
        banned = {frozenset((n0[y], n0[y + 1]))}
        res = greedy_route(g, s, t, banned)
        if res is None:
            continue
        key = tuple(res[1])
        if key in seen:
            continue
        seen.add(key)
        routes.append(_route_from(g, *res, label=f"X{y + 1} (sin conexión {y + 1})"))
    best = int(np.argmin([r.length for r in routes]))
    return routes, best


def dijkstra_route(g: SkeletonGraph, s: int, t: int) -> Route | None:
    dist = {s: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == t:
            break
        if d > dist.get(u, np.inf):
            continue
        for eid in g.adj[u]:
            v = g.other(eid, u)
            nd = d + g.edges[eid].length
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = (u, eid)
                heapq.heappush(heap, (nd, v))
    if t not in dist:
        return None
    nodes = [t]
    edges = []
    while nodes[-1] != s:
        u, eid = prev[nodes[-1]]
        edges.append(eid)
        nodes.append(u)
    return _route_from(g, nodes[::-1], edges[::-1], label="Dijkstra (óptima en el grafo)")


def plan_route(g: SkeletonGraph, s: int, t: int, method: str = "paper") -> tuple[list[Route], int]:
    if method == "paper":
        return paper_routes(g, s, t)
    if method == "dijkstra":
        r = dijkstra_route(g, s, t)
        return ([r], 0) if r is not None else ([], -1)
    raise ValueError("route_method debe ser 'paper' o 'dijkstra'")
