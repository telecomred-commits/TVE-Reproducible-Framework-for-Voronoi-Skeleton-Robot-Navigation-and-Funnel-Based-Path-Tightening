"""Planificación por muestreo: RRT, RRT* y PRM.

* **RRT** (LaValle, 1998): árbol que crece desde S hacia puntos aleatorios del
  espacio libre (con sesgo hacia T).  Probabilísticamente completo, rápido en
  espacios amplios, pero sus rutas son quebradas y no óptimas.
* **RRT*** (Karaman y Frazzoli, 2011): al insertar cada nodo elige el mejor
  padre entre sus vecinos y «recablea» el árbol; asintóticamente óptimo.
* **PRM** (Kavraki et al., 1996): hoja de ruta probabilística — muestras en el
  espacio libre unidas con sus k vecinos más cercanos si hay línea de vista;
  luego Dijkstra.  Multi-consulta: el grafo se puede reutilizar.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from common.geometry import segment_free
from common.problem import AlgorithmResult, Problem


def _free_sampler(problem: Problem, rng):
    ys, xs = np.nonzero(~problem.occ)
    pts = np.stack([xs, ys], axis=1).astype(float)

    def sample(goal_bias, goal):
        if rng.random() < goal_bias:
            return goal.copy()
        p = pts[rng.integers(len(pts))]
        return p + rng.uniform(-0.5, 0.5, 2)
    return sample, pts


def _backtrack(nodes, parent, i):
    out = [i]
    while parent[out[-1]] >= 0:
        out.append(parent[out[-1]])
    return nodes[out[::-1]]


def run_rrt(problem: Problem, paso_m=0.5, iteraciones=6000, sesgo_meta=0.08, semilla=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(semilla)
    step = max(2.0, paso_m / problem.mpp)
    sample, _ = _free_sampler(problem, rng)
    S, T = problem.start, problem.goal
    nodes = np.zeros((iteraciones + 2, 2))
    parent = np.full(iteraciones + 2, -1, dtype=np.int64)
    nodes[0] = S
    n = 1
    goal_idx = -1
    it = 0
    if segment_free(problem.occ, S, T):
        nodes[1] = T
        parent[1] = 0
        n = 2
        goal_idx = 1
    while goal_idx < 0 and it < iteraciones:
        it += 1
        q = sample(sesgo_meta, T)
        d = np.hypot(nodes[:n, 0] - q[0], nodes[:n, 1] - q[1])
        i = int(np.argmin(d))
        if d[i] < 1e-9:
            continue
        new = nodes[i] + (q - nodes[i]) * min(1.0, step / d[i])
        if not segment_free(problem.occ, nodes[i], new):
            continue
        nodes[n] = new
        parent[n] = i
        n += 1
        if np.hypot(*(new - T)) <= step and segment_free(problem.occ, new, T):
            nodes[n] = T
            parent[n] = n - 1
            goal_idx = n
            n += 1
    edges = np.stack([nodes[parent[1:n]], nodes[1:n]], axis=1) if n > 1 else np.zeros((0, 2, 2))
    stats = {"iteraciones": it, "nodos_arbol": n}
    viz = {"tree": edges}
    if goal_idx < 0:
        return AlgorithmResult("rrt", "RRT", False, None, time.perf_counter() - t0,
                               "No se alcanzó T en el número de iteraciones", stats, viz)
    path = _backtrack(nodes[:n], parent[:n], goal_idx)
    return AlgorithmResult("rrt", "RRT", True, path, time.perf_counter() - t0, "Ruta encontrada", stats, viz)


def run_rrt_star(problem: Problem, paso_m=0.5, iteraciones=4000, sesgo_meta=0.05, gamma=2.5, semilla=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(semilla)
    step = max(2.0, paso_m / problem.mpp)
    sample, free_pts = _free_sampler(problem, rng)
    S, T = problem.start, problem.goal
    cap = iteraciones + 2
    nodes = np.zeros((cap, 2))
    parent = np.full(cap, -1, dtype=np.int64)
    cost = np.zeros(cap)
    children: list[list[int]] = [[] for _ in range(cap)]
    nodes[0] = S
    n = 1
    goal_parent = -1
    goal_cost = math.inf
    first_solution_it = None
    # radio de conexión: γ·(log n / n)^(1/2) escalado al área libre
    area = len(free_pts)
    r_scale = gamma * math.sqrt(area / math.pi)
    it = 0
    for it in range(1, iteraciones + 1):
        q = sample(sesgo_meta, T)
        d = np.hypot(nodes[:n, 0] - q[0], nodes[:n, 1] - q[1])
        i = int(np.argmin(d))
        if d[i] < 1e-9:
            continue
        new = nodes[i] + (q - nodes[i]) * min(1.0, step / d[i])
        if not segment_free(problem.occ, nodes[i], new):
            continue
        radius = max(step * 1.5, min(r_scale * math.sqrt(math.log(n + 1) / (n + 1)), step * 4))
        dn = np.hypot(nodes[:n, 0] - new[0], nodes[:n, 1] - new[1])
        near = np.flatnonzero(dn <= radius)
        best, best_c = i, cost[i] + np.hypot(*(new - nodes[i]))
        free_near = {i: True}
        for j in near[np.argsort(cost[near] + dn[near])]:
            c = cost[j] + dn[j]
            if c >= best_c:
                break
            if segment_free(problem.occ, nodes[j], new):
                best, best_c = int(j), c
                free_near[int(j)] = True
                break
        k = n
        nodes[k] = new
        parent[k] = best
        cost[k] = best_c
        children[best].append(k)
        n += 1
        # recableado
        for j in near:
            j = int(j)
            if j == best:
                continue
            c = best_c + dn[j]
            if c + 1e-9 < cost[j] and segment_free(problem.occ, new, nodes[j]):
                old = parent[j]
                children[old].remove(j)
                parent[j] = k
                children[k].append(j)
                delta = cost[j] - c
                stack = [j]
                while stack:              # propagar la mejora al subárbol
                    u = stack.pop()
                    cost[u] -= delta
                    stack.extend(children[u])
        # conexión con la meta
        dT = np.hypot(*(new - T))
        if dT <= step * 1.5 and cost[k] + dT < goal_cost and segment_free(problem.occ, new, T):
            goal_parent, goal_cost = k, cost[k] + dT
            if first_solution_it is None:
                first_solution_it = it
    # la mejor conexión a T puede haber mejorado por recableado
    if goal_parent >= 0:
        cand = np.flatnonzero(np.hypot(nodes[:n, 0] - T[0], nodes[:n, 1] - T[1]) <= step * 1.5)
        for j in cand[np.argsort(cost[cand])]:
            c = cost[j] + np.hypot(*(nodes[j] - T))
            if c < goal_cost and segment_free(problem.occ, nodes[j], T):
                goal_parent, goal_cost = int(j), c
    edges = np.stack([nodes[parent[1:n]], nodes[1:n]], axis=1) if n > 1 else np.zeros((0, 2, 2))
    stats = {"iteraciones": it, "nodos_arbol": n, "primera_solucion_iter": first_solution_it}
    viz = {"tree": edges}
    if goal_parent < 0:
        return AlgorithmResult("rrt_star", "RRT*", False, None, time.perf_counter() - t0,
                               "No se alcanzó T en el número de iteraciones", stats, viz)
    path = np.vstack([_backtrack(nodes[:n], parent[:n], goal_parent), T])
    return AlgorithmResult("rrt_star", "RRT*", True, path, time.perf_counter() - t0, "Ruta encontrada", stats, viz)


def run_prm(problem: Problem, muestras=600, vecinos=12, semilla=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(semilla)
    ys, xs = np.nonzero(~problem.occ)
    idx = rng.choice(len(ys), size=min(muestras, len(ys)), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(float)
    pts = np.vstack([problem.start, problem.goal, pts])
    n = len(pts)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    edges = []
    checked = set()
    k = min(vecinos, n - 1)
    D = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    for i in range(n):
        for j in np.argsort(D[i])[1:k + 1]:
            j = int(j)
            key = (min(i, j), max(i, j))
            if key in checked:
                continue
            checked.add(key)
            if segment_free(problem.occ, pts[i], pts[j]):
                adj[i].append((j, D[i, j]))
                adj[j].append((i, D[i, j]))
                edges.append((pts[i], pts[j]))
    # Dijkstra de S (0) a T (1)
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
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (d + w, v))
    stats = {"muestras": n, "aristas": len(edges), "chequeos_colision": len(checked)}
    viz = {"graph": np.asarray(edges) if edges else np.zeros((0, 2, 2)), "nodes": pts}
    if not math.isfinite(dist[1]):
        return AlgorithmResult("prm", "PRM", False, None, time.perf_counter() - t0,
                               "La hoja de ruta no conecta S y T (aumente muestras o vecinos)", stats, viz)
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    return AlgorithmResult("prm", "PRM", True, pts[path[::-1]], time.perf_counter() - t0, "Ruta encontrada",
                           stats, viz)
