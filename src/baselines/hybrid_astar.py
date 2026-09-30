"""Hybrid A* (Dolgov, Thrun, Montemerlo y Diebel, 2008) — robots NO holonómicos.

Busca en el espacio de estados continuo (x, y, θ) de un vehículo tipo
Ackermann (auto) que no puede girar sobre sí mismo: su curvatura está acotada
por |κ| ≤ 1/R_min.  Cada expansión aplica primitivas de movimiento (arcos de
longitud fija con varias curvaturas, hacia adelante y opcionalmente en
reversa).  Los estados se agrupan en celdas (x, y, θ) discretas para no
re-expandir, pero conservan su posición continua, así que las rutas son
cinemáticamente factibles por construcción.

Heurística: máximo entre la distancia euclidiana y la distancia geodésica
holonómica con obstáculos (Dijkstra inverso desde T), ambas admisibles.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from common.geometry import polyline_free
from common.problem import AlgorithmResult, Problem
from baselines.grid_search import distance_field


def _arc(x, y, th, kappa, L, n):
    s = np.linspace(0.0, L, n)
    if abs(kappa) < 1e-9:
        return x + s * np.cos(th), y + s * np.sin(th), np.full(n, th)
    t = th + kappa * s
    return x + (np.sin(t) - math.sin(th)) / kappa, y - (np.cos(t) - math.cos(th)) / kappa, t


def run_hybrid_astar(problem: Problem, radio_giro_m=1.0, paso_m=0.35, direcciones=72, curvaturas=5,
                     reversa=False, orientacion_inicial="hacia T", max_expansiones=60000):
    t0 = time.perf_counter()
    mpp = problem.mpp
    R = max(radio_giro_m / mpp, 1.0)
    L = max(paso_m / mpp, 2.0)
    kmax = 1.0 / R
    kappas = np.linspace(-kmax, kmax, max(3, int(curvaturas) | 1))
    xy_cell = max(1.0, L * 0.7)
    n_th = int(direcciones)
    S, T = problem.start, problem.goal
    if orientacion_inicial == "hacia T":
        th0 = math.atan2(T[1] - S[1], T[0] - S[0])
    else:
        th0 = math.radians({"este (0°)": 0, "sur (90°)": 90, "oeste (180°)": 180, "norte (270°)": 270}
                           .get(orientacion_inicial, 0))
    D, dcell = distance_field(problem.occ, T, cell=max(2, int(round(xy_cell / 2))))
    Hh, Wh = D.shape

    def heur(x, y):
        r = min(max(int((y + 0.5) // dcell), 0), Hh - 1)
        c = min(max(int((x + 0.5) // dcell), 0), Wh - 1)
        g = D[r, c]
        e = math.hypot(x - T[0], y - T[1])
        return max(e, g - 1.5 * dcell) if math.isfinite(g) else e

    def key(x, y, th):
        return (int(x // xy_cell), int(y // xy_cell), int(round((th % (2 * math.pi)) / (2 * math.pi) * n_th)) % n_th)

    dirs = [1.0, -1.0] if reversa else [1.0]
    n_samp = max(3, int(L) + 1)
    nodes = [(S[0], S[1], th0)]
    parent = [-1]
    arcs = [np.array([[S[0], S[1]]])]
    gcost = [0.0]
    best_g = {key(*nodes[0]): 0.0}
    heap = [(heur(S[0], S[1]), 0.0, 0)]
    closed = set()
    expanded = 0
    goal_node = -1
    final_shot = None
    while heap and expanded < max_expansiones:
        f, g, i = heapq.heappop(heap)
        x, y, th = nodes[i]
        k = key(x, y, th)
        if k in closed:
            continue
        closed.add(k)
        expanded += 1
        # Expansión analítica: un único arco tangente al rumbo actual que pasa por T.
        # Su curvatura es κ = 2·sen(α)/d; es factible si |κ| ≤ 1/R y no choca.
        dT = math.hypot(x - T[0], y - T[1])
        if dT <= 6 * L:
            alpha = (math.atan2(T[1] - y, T[0] - x) - th + math.pi) % (2 * math.pi) - math.pi
            if abs(alpha) < math.pi / 2:
                kap = 2 * math.sin(alpha) / max(dT, 1e-9)
                if abs(kap) <= kmax + 1e-12:
                    s_len = dT if abs(alpha) < 1e-9 else dT * alpha / math.sin(alpha)
                    ax, ay, _ = _arc(x, y, th, kap, s_len, max(3, int(s_len) + 1))
                    shot = np.stack([ax, ay], axis=1)
                    if polyline_free(problem.occ, shot):
                        shot[-1] = T
                        goal_node = i
                        final_shot = shot
                        break
        for d in dirs:
            for kap in kappas:
                ax, ay, at = _arc(x, y, th, kap, d * L, n_samp)
                pts = np.stack([ax, ay], axis=1)
                if not polyline_free(problem.occ, pts):
                    continue
                nx, ny, nth = ax[-1], ay[-1], at[-1]
                nk = key(nx, ny, nth)
                if nk in closed:
                    continue
                cost = L * (1.0 if d > 0 else 2.0) + 0.1 * L * abs(kap) / kmax
                ng = g + cost
                if ng >= best_g.get(nk, math.inf):
                    continue
                best_g[nk] = ng
                nodes.append((nx, ny, nth))
                parent.append(i)
                arcs.append(pts)
                gcost.append(ng)
                heapq.heappush(heap, (ng + heur(nx, ny), ng, len(nodes) - 1))
    stats = {"estados_expandidos": expanded, "estados_generados": len(nodes),
             "radio_giro_px": R, "longitud_primitiva_px": L}
    viz = {"states": np.array([[n[0], n[1]] for n in nodes[:: max(1, len(nodes) // 4000)]])}
    if goal_node < 0:
        return AlgorithmResult("hybrid_astar", "Hybrid A*", False, None, time.perf_counter() - t0,
                               "No se encontró una ruta factible con ese radio de giro", stats, viz)
    chain = [goal_node]
    while parent[chain[-1]] >= 0:
        chain.append(parent[chain[-1]])
    chain.reverse()
    pts = [arcs[chain[0]]]
    for i in chain[1:]:
        pts.append(arcs[i][1:])
    path = np.vstack(pts + [final_shot[1:]])
    stats["final_heading_deg"] = math.degrees(nodes[goal_node][2]) % 360
    return AlgorithmResult("hybrid_astar", "Hybrid A*", True, path, time.perf_counter() - t0,
                           "Ruta factible para vehículo tipo Ackermann", stats, viz)
