"""Optimización por colonia de hormigas (ACO; Dorigo, 1992; Ant System elitista).

Sobre una malla de ocupación, m hormigas parten de S y en cada paso eligen una
celda vecina libre y no visitada con probabilidad

    p(i→j) ∝ τ_j^α · η_j^β ,   η_j = exp(d(i, T) − d(j, T))   (avance hacia T)

Las hormigas que llegan a T depositan feromona Q/L en su recorrido; la
feromona se evapora (factor ρ) en cada iteración y la mejor ruta encontrada
deposita un refuerzo extra (elitismo).  Con las iteraciones la colonia
converge hacia rutas cortas.  Es una metaheurística: no garantiza optimalidad
ni completitud y su resultado depende de la semilla.

Todas las hormigas avanzan en paralelo (operaciones vectorizadas).
"""
from __future__ import annotations

import time

import numpy as np

from common.geometry import segment_free
from common.problem import AlgorithmResult, Problem, cell_center, coarse_grid, point_cell
from baselines.grid_search import _attach

SQ2 = np.sqrt(2.0)


def _prune(seq: np.ndarray, Wp: int, blk: np.ndarray) -> np.ndarray:
    """Poda de rodeos: desde cada celda salta a la ÚLTIMA celda posterior del
    recorrido que sea vecina suya (movimiento válido).  Elimina los lazos y
    desvíos que deja la exploración con retroceso."""
    seq = [int(v) for v in seq]
    where: dict[int, int] = {}
    for k, v in enumerate(seq):
        where[v] = k                                   # última aparición
    out = [seq[0]]
    i = 0
    n = len(seq)
    offs = (-Wp, Wp, -1, 1, -Wp - 1, -Wp + 1, Wp - 1, Wp + 1)
    while i < n - 1:
        u = seq[i]
        best = i + 1
        for off in offs:
            j = where.get(u + off, -1)
            if j > best:
                if off in (-Wp - 1, -Wp + 1, Wp - 1, Wp + 1):
                    dr = -Wp if off < 0 else Wp
                    dc = off - dr
                    if blk[u + dr] or blk[u + dc]:
                        continue
                best = j
        out.append(seq[best])
        i = best
    return np.asarray(out, dtype=np.int64)


def run_aco(problem: Problem, celda=4, hormigas=40, iteraciones=40, alfa=1.0, beta=2.0, evaporacion=0.25,
            elitismo=2.0, semilla=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(semilla)
    cell = max(1, int(celda))
    grid = coarse_grid(problem.occ, cell)
    H, W = grid.shape
    s = _attach(problem, grid, cell, problem.start)
    g = _attach(problem, grid, cell, problem.goal)
    if s is None or g is None:
        return AlgorithmResult("aco", "Colonia de hormigas", False, None, time.perf_counter() - t0,
                               "S o T no se pueden conectar a la malla")
    Wp = W + 2
    blocked = np.ones((H + 2, W + 2), bool)
    blocked[1:-1, 1:-1] = grid
    blk = blocked.ravel()
    N = blk.size
    si = (s[0] + 1) * Wp + s[1] + 1
    gi = (g[0] + 1) * Wp + g[1] + 1
    rr, cc = np.divmod(np.arange(N), Wp)
    dgoal = np.hypot(rr - (g[0] + 1), cc - (g[1] + 1))            # distancia a T [celdas]
    Q = max(float(dgoal[si]), 1.0)                                  # depósito ~ 1 para una ruta recta
    offs = np.array([-Wp, Wp, -1, 1, -Wp - 1, -Wp + 1, Wp - 1, Wp + 1])
    step_len = np.array([1, 1, 1, 1, SQ2, SQ2, SQ2, SQ2])
    # diagonales sin cortar esquinas: requieren libres los dos ortogonales
    diag_req = {4: (0, 2), 5: (0, 3), 6: (1, 2), 7: (1, 3)}
    tau = np.ones(N)
    m = int(hormigas)
    n_free = int((~grid).sum())
    max_steps = int(min(2 * n_free + 10, 12 * (H + W) + 50))
    best_path, best_len = None, np.inf
    curve = []
    successes = 0
    rows = np.arange(m)
    for it in range(int(iteraciones)):
        # Cada hormiga guarda su recorrido como una pila: si queda atrapada en
        # un callejón (todas las vecinas visitadas u ocupadas) retrocede una
        # celda.  La pila final es un camino simple S -> T sin callejones.
        stack = np.zeros((m, max_steps + 2), dtype=np.int64)
        stack[:, 0] = si
        sp = np.ones(m, dtype=np.int64)
        pos = np.full(m, si)
        alive = np.ones(m, bool)
        arrived = np.zeros(m, bool)
        visited = np.zeros((m, N), bool)
        visited[:, si] = True
        for _ in range(max_steps):
            act = alive & ~arrived
            if not act.any():
                break
            nb = pos[:, None] + offs[None, :]                      # (m, 8)
            ok = ~blk[nb] & ~visited[rows[:, None], nb]
            for k, (a, b) in diag_req.items():
                ok[:, k] &= ~blk[pos + offs[a]] & ~blk[pos + offs[b]]
            # heurística de avance: η = exp(d(actual) - d(vecina)); la versión clásica
            # 1/d casi no distingue vecinos cuando T está lejos.
            eta = np.exp(np.clip(dgoal[pos][:, None] - dgoal[nb], -2, 2))
            w = (tau[nb] ** alfa) * (eta ** beta) * ok
            tot = w.sum(axis=1)
            stuck = act & (tot <= 0)
            if stuck.any():                                         # retroceso
                sidx = np.flatnonzero(stuck)
                sp[sidx] -= 1
                gone = sidx[sp[sidx] <= 0]
                alive[gone] = False
                back = sidx[sp[sidx] > 0]
                pos[back] = stack[back, sp[back] - 1]
            mv = act & ~stuck
            if not mv.any():
                continue
            idx = np.flatnonzero(mv)
            prob = w[idx] / tot[idx, None]
            u = rng.random(len(idx))[:, None]
            choice = (prob.cumsum(axis=1) < u).sum(axis=1).clip(0, 7)
            newp = nb[idx, choice]
            pos[idx] = newp
            stack[idx, sp[idx]] = newp
            sp[idx] += 1
            visited[idx, newp] = True
            arrived[idx] |= newp == gi
        tau *= (1.0 - evaporacion)
        for a in np.flatnonzero(arrived):
            seq = _prune(stack[a, :sp[a]], Wp, blk)
            r_, c_ = np.divmod(seq, Wp)
            L = float(np.hypot(np.diff(r_), np.diff(c_)).sum())
            tau[seq] += Q / max(L, 1e-9)
            successes += 1
            if L < best_len:
                best_len, best_path = L, seq.copy()
        if best_path is not None and elitismo > 0:
            tau[best_path] += elitismo * Q / best_len
        tau = np.clip(tau, 1e-3, 1e3)
        curve.append(best_len * cell if np.isfinite(best_len) else np.nan)
    tau_img = tau.reshape(H + 2, Wp)[1:-1, 1:-1]
    stats = {"iteraciones": int(iteraciones), "hormigas": m, "hormigas_que_llegaron": successes,
             "celda_px": cell}
    viz = {"pheromone": tau_img, "cell": cell, "curve": np.asarray(curve)}
    if best_path is None:
        return AlgorithmResult("aco", "Colonia de hormigas", False, None, time.perf_counter() - t0,
                               "Ninguna hormiga llegó a T", stats, viz)
    pts = [problem.start] + [cell_center(i // Wp - 1, i % Wp - 1, cell) for i in best_path] + [problem.goal]
    path = np.asarray(pts, float)
    return AlgorithmResult("aco", "Colonia de hormigas", True, path, time.perf_counter() - t0,
                           "Ruta encontrada", stats, viz)
