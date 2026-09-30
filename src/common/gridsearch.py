"""Búsqueda A* en malla 8-conexa.

No forma parte del método del artículo; se usa como referencia en las
simulaciones (longitud casi óptima tras "string pulling") y como respaldo si
S o T no pueden conectarse al esqueleto.
"""
from __future__ import annotations

import heapq

import numpy as np

from common.geometry import segment_free

_SQ2 = 1.4142135623730951


def astar(occ: np.ndarray, start_xy, goal_xy) -> np.ndarray | None:
    h, w = occ.shape
    s = (int(round(start_xy[1])), int(round(start_xy[0])))
    g = (int(round(goal_xy[1])), int(round(goal_xy[0])))
    for r, c in (s, g):
        if not (0 <= r < h and 0 <= c < w) or occ[r, c]:
            return None
    free = ~occ
    gr, gc = g

    def heur(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (_SQ2 - 2) * min(dr, dc)

    dist = np.full((h, w), np.inf)
    dist[s] = 0.0
    parent = -np.ones((h, w), dtype=np.int64)
    closed = np.zeros((h, w), dtype=bool)
    heap = [(heur(*s), 0.0, s[0], s[1])]
    nbrs = [(-1, -1, _SQ2), (-1, 0, 1.0), (-1, 1, _SQ2), (0, -1, 1.0),
            (0, 1, 1.0), (1, -1, _SQ2), (1, 0, 1.0), (1, 1, _SQ2)]
    found = False
    while heap:
        _, d, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == g:
            found = True
            break
        for dr, dc, cost in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not closed[rr, cc]:
                # evita cortar esquinas en diagonal entre dos obstáculos
                if dr and dc and (occ[r, cc] and occ[rr, c]):
                    continue
                nd = d + cost
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    parent[rr, cc] = r * w + c
                    heapq.heappush(heap, (nd + heur(rr, cc), nd, rr, cc))
    if not found:
        return None
    path = [g]
    while path[-1] != s:
        p = int(parent[path[-1]])
        path.append(divmod(p, w))
    path = np.asarray([(c, r) for r, c in path[::-1]], dtype=float)
    path[0] = np.asarray(start_xy, dtype=float)
    path[-1] = np.asarray(goal_xy, dtype=float)
    return path


def string_pull(occ: np.ndarray, path: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Atajos por línea de vista (aproximación a la ruta euclidiana más corta)."""
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path.copy()
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        # búsqueda exponencial + binaria del punto visible más lejano
        lo, hi = i + 1, i + 1
        stride = 1
        while hi < n - 1 and segment_free(occ, path[i], path[min(hi + stride, n - 1)], step):
            lo = min(hi + stride, n - 1)
            hi = lo
            stride *= 2
        top = min(hi + stride, n - 1)
        a, b = lo, top
        while b - a > 1:
            m = (a + b) // 2
            if segment_free(occ, path[i], path[m], step):
                a = m
            else:
                b = m
        j = a if not segment_free(occ, path[i], path[b], step) else b
        j = max(j, i + 1)
        out.append(path[j])
        i = j
    return np.asarray(out)
