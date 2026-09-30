"""Búsqueda en malla: Dijkstra, A* y Theta*.

* **Dijkstra** (1959): expande los nodos en orden de costo acumulado g(n).
  Completo y óptimo sobre la malla, pero explora en todas las direcciones.
* **A*** (Hart, Nilsson y Raphael, 1968): ordena por f(n) = g(n) + w·h(n) con la
  heurística octil h, admisible y consistente en una malla 8-conexa.  Con w = 1
  encuentra la misma ruta óptima que Dijkstra expandiendo muchos menos nodos;
  con w > 1 (A* ponderado) es más rápido pero hasta w veces más largo.
* **Theta*** (Nash et al., 2007): A* de ángulo libre.  Al expandir un vecino
  intenta conectarlo directamente con el padre de su padre si hay línea de
  vista, así que las rutas no están atadas a los 8 rumbos de la malla.

La malla se construye a partir del espacio de configuración con celdas de
``celda`` píxeles (conservadora: una celda con cualquier píxel ocupado está
ocupada).  No se permite cortar esquinas en diagonal.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from common.geometry import segment_free
from common.problem import AlgorithmResult, Problem, cell_center, coarse_grid, point_cell

SQ2 = math.sqrt(2.0)


def _attach(problem: Problem, grid, cell, p):
    """Celda libre de la malla conectada por una recta libre al punto p."""
    H, W = grid.shape
    r, c = point_cell(p, cell, grid.shape)
    if not grid[r, c]:
        return r, c
    for rad in range(1, 12):
        cands = []
        for rr in range(r - rad, r + rad + 1):
            for cc in range(c - rad, c + rad + 1):
                if 0 <= rr < H and 0 <= cc < W and not grid[rr, cc] and max(abs(rr - r), abs(cc - c)) == rad:
                    cands.append((np.hypot(*(cell_center(rr, cc, cell) - p)), rr, cc))
        for _, rr, cc in sorted(cands):
            if segment_free(problem.occ, p, cell_center(rr, cc, cell)):
                return rr, cc
    return None


def grid_search(problem: Problem, mode: str = "astar", cell: int = 2, weight: float = 1.0,
                diagonal: bool = True, max_expansions: int = 5_000_000) -> AlgorithmResult:
    t0 = time.perf_counter()
    cell = max(1, int(cell))
    grid = coarse_grid(problem.occ, cell)
    H, W = grid.shape
    s = _attach(problem, grid, cell, problem.start)
    g_ = _attach(problem, grid, cell, problem.goal)
    names = {"dijkstra": "Dijkstra", "astar": "A*", "theta": "Theta*"}
    if s is None or g_ is None:
        return AlgorithmResult(mode, names[mode], False, None, time.perf_counter() - t0,
                               "S o T no se pueden conectar a la malla (aumente la resolución)")
    # malla con borde ocupado para evitar comprobaciones de límites
    Wp = W + 2
    blocked = np.ones((H + 2, W + 2), bool)
    blocked[1:-1, 1:-1] = grid
    blk = blocked.ravel().tolist()
    si = (s[0] + 1) * Wp + (s[1] + 1)
    gi = (g_[0] + 1) * Wp + (g_[1] + 1)
    gr, gc = g_[0] + 1, g_[1] + 1
    if diagonal:
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]
    else:
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    moves = [(dr * Wp + dc, dr, dc, cost) for dr, dc, cost in moves]
    N = (H + 2) * Wp
    INF = float("inf")
    gcost = [INF] * N
    parent = [-1] * N
    closed = bytearray(N)
    gcost[si] = 0.0
    parent[si] = si

    use_h = mode in ("astar", "theta")
    wgt = weight if mode == "astar" else 1.0

    def heur(i):
        r, c = divmod(i, Wp)
        dr, dc = abs(r - gr), abs(c - gc)
        if mode == "theta":
            return math.hypot(dr, dc)
        if diagonal:
            return (dr + dc) + (SQ2 - 2) * min(dr, dc)
        return dr + dc

    def center(i):
        r, c = divmod(i, Wp)
        return cell_center(r - 1, c - 1, cell)

    los_cache = {}

    def los(a, b):
        key = (a, b) if a < b else (b, a)
        v = los_cache.get(key)
        if v is None:
            v = segment_free(problem.occ, center(a), center(b))
            los_cache[key] = v
        return v

    heap = [(wgt * heur(si) if use_h else 0.0, 0.0, si)]
    expanded = 0
    found = False
    while heap:
        f, g, u = heapq.heappop(heap)
        if closed[u]:
            continue
        closed[u] = 1
        expanded += 1
        if u == gi:
            found = True
            break
        if expanded > max_expansions:
            break
        ur, uc = divmod(u, Wp)
        for off, dr, dc, cost in moves:
            v = u + off
            if blk[v] or closed[v]:
                continue
            if dr and dc and (blk[u + dr * Wp] or blk[u + dc]):
                continue                      # sin cortar esquinas
            if mode == "theta":
                pu = parent[u]
                if pu != u and los(pu, v):
                    pr, pc = divmod(pu, Wp)
                    vr, vc = divmod(v, Wp)
                    ng = gcost[pu] + math.hypot(vr - pr, vc - pc)
                    if ng < gcost[v]:
                        gcost[v] = ng
                        parent[v] = pu
                        heapq.heappush(heap, (ng + heur(v), ng, v))
                    continue
            ng = g + cost
            if ng < gcost[v]:
                gcost[v] = ng
                parent[v] = u
                heapq.heappush(heap, (ng + (wgt * heur(v) if use_h else 0.0), ng, v))
    visited = np.frombuffer(bytes(closed), dtype=np.uint8).reshape(H + 2, Wp)[1:-1, 1:-1].astype(bool)
    stats = {"nodos_expandidos": expanded, "celdas_malla": int(H * W), "celda_px": cell}
    viz = {"visited": visited, "cell": cell}
    if not found:
        return AlgorithmResult(mode, names[mode], False, None, time.perf_counter() - t0,
                               "La búsqueda no encontró ruta", stats, viz)
    idx = [gi]
    while idx[-1] != si:
        idx.append(parent[idx[-1]])
    idx.reverse()
    pts = [problem.start] + [center(i) for i in idx] + [problem.goal]
    path = _dedupe(np.asarray(pts, float))
    stats["costo_malla_px"] = gcost[gi] * cell
    return AlgorithmResult(mode, names[mode], True, path, time.perf_counter() - t0, "Ruta encontrada", stats, viz)


def _dedupe(path):
    keep = [0]
    for i in range(1, len(path)):
        if np.hypot(*(path[i] - path[keep[-1]])) > 1e-9:
            keep.append(i)
    return path[keep]


def run_dijkstra(problem, celda=2, diagonal=True):
    return grid_search(problem, "dijkstra", cell=celda, diagonal=diagonal)


def run_astar(problem, celda=2, peso=1.0, diagonal=True):
    return grid_search(problem, "astar", cell=celda, weight=peso, diagonal=diagonal)


def run_theta(problem, celda=2):
    return grid_search(problem, "theta", cell=celda)


def distance_field(occ: np.ndarray, goal, cell: int = 2) -> tuple[np.ndarray, int]:
    """Distancia geodésica (Dijkstra inverso) desde la meta a todas las celdas [px].

    Se usa como heurística holonómica con obstáculos en Hybrid A*.
    """
    grid = coarse_grid(occ, cell)
    H, W = grid.shape
    gr, gc = point_cell(goal, cell, grid.shape)
    Wp = W + 2
    blocked = np.ones((H + 2, W + 2), bool)
    blocked[1:-1, 1:-1] = grid
    blk = blocked.ravel().tolist()
    N = (H + 2) * Wp
    dist = [math.inf] * N
    start = (gr + 1) * Wp + gc + 1
    dist[start] = 0.0
    heap = [(0.0, start)]
    moves = [(-Wp, 1.0), (Wp, 1.0), (-1, 1.0), (1, 1.0),
             (-Wp - 1, SQ2), (-Wp + 1, SQ2), (Wp - 1, SQ2), (Wp + 1, SQ2)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for off, cost in moves:
            v = u + off
            if blk[v]:
                continue
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    D = np.asarray(dist, float).reshape(H + 2, Wp)[1:-1, 1:-1] * cell
    return D, cell
