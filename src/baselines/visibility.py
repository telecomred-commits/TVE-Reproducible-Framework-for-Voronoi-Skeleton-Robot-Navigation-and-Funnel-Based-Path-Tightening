"""Grafo de visibilidad (Lozano-Pérez y Wesley, 1979).

Para obstáculos poligonales, la ruta euclidiana más corta en el plano es una
poligonal cuyos vértices interiores son vértices convexos de los obstáculos.
Se aproximan los contornos del espacio de configuración con polígonos, se
toman sus vértices convexos (desplazados 1.5 px hacia el espacio libre), se
unen todos los pares mutuamente visibles y se aplica Dijkstra.

Da la ruta óptima (salvo la aproximación poligonal de 1 px), por eso se usa
como **referencia** de longitud en la comparación.  Sus rutas rozan los
obstáculos (holgura mínima) y el costo crece como O(V² · chequeo).
"""
from __future__ import annotations

import heapq
import math
import time

import cv2
import numpy as np

from common.geometry import points_in_collision, segment_free
from common.problem import AlgorithmResult, Problem


def convex_vertices(occ: np.ndarray, eps: float = 1.0, offset: float = 2.0) -> np.ndarray:
    cnts, hier = cv2.findContours(occ.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    h, w = occ.shape
    for cnt in cnts:
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(float)
        n = len(poly)
        if n < 3:
            continue
        for i in range(n):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
            e1 = b - a
            e2 = c - b
            if np.hypot(*e1) < 1e-9 or np.hypot(*e2) < 1e-9:
                continue
            u1 = e1 / np.hypot(*e1)
            u2 = e2 / np.hypot(*e2)
            # La bisectriz u2 - u1 apunta hacia el lado del ángulo menor.  En una
            # esquina CONVEXA del obstáculo ese lado es el obstáculo, así que el
            # vértice útil está en -bisectriz (espacio libre).  En las esquinas
            # cóncavas el lado del ángulo menor es libre: no sirven para rutas cortas.
            bis = u2 - u1
            nb = np.hypot(*bis)
            if nb < 1e-6:
                continue
            bis /= nb
            ahead = b + 1.5 * bis
            if not (0 <= ahead[0] < w and 0 <= ahead[1] < h) or not points_in_collision(occ, ahead[None])[0]:
                continue                                 # no es esquina convexa
            for off in (offset, offset + 1.0, offset + 2.5):
                q = b - off * bis
                if 0 <= q[0] < w and 0 <= q[1] < h and not points_in_collision(occ, q[None])[0]:
                    out.append(q)
                    break
    if not out:
        return np.zeros((0, 2))
    pts = np.asarray(out)
    # eliminar duplicados cercanos
    keep = []
    for p in pts:
        if all(np.hypot(*(p - k)) > 1.0 for k in keep):
            keep.append(p)
    return np.asarray(keep)


def run_visibility(problem: Problem, tolerancia_poligono=1.0):
    t0 = time.perf_counter()
    V = convex_vertices(problem.occ, eps=tolerancia_poligono)
    pts = np.vstack([problem.start, problem.goal, V]) if len(V) else np.vstack([problem.start, problem.goal])
    n = len(pts)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    edges = []
    checks = 0
    for i in range(n):
        for j in range(i + 1, n):
            checks += 1
            if segment_free(problem.occ, pts[i], pts[j]):
                d = float(np.hypot(*(pts[i] - pts[j])))
                adj[i].append((j, d))
                adj[j].append((i, d))
                edges.append((pts[i], pts[j]))
    dist = [math.inf] * n
    prev = [-1] * n
    dist[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == 1:
            break
        if d > dist[u]:
            continue
        for v, wgt in adj[u]:
            if d + wgt < dist[v]:
                dist[v] = d + wgt
                prev[v] = u
                heapq.heappush(heap, (d + wgt, v))
    stats = {"vertices": n, "aristas_visibles": len(edges), "chequeos_colision": checks}
    viz = {"graph": np.asarray(edges) if edges else np.zeros((0, 2, 2)), "nodes": pts}
    if not math.isfinite(dist[1]):
        return AlgorithmResult("visibility", "Grafo de visibilidad", False, None, time.perf_counter() - t0,
                               "S y T no están conectados en el grafo de visibilidad", stats, viz)
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    return AlgorithmResult("visibility", "Grafo de visibilidad", True, pts[path[::-1]], time.perf_counter() - t0,
                           "Ruta encontrada", stats, viz)
