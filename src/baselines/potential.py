"""Campos de potencial artificial (Khatib, 1986).

El robot desciende por el gradiente de U = U_atr + U_rep:

* U_atr = ½ ξ d²  (cónico lejos de la meta, para no acelerar sin límite)
* U_rep = ½ η (1/ρ − 1/ρ₀)²  si ρ < ρ₀, con ρ la distancia al obstáculo

Es un método **reactivo**: muy barato y suave, pero no completo — queda
atrapado en mínimos locales (obstáculos cóncavos, pasos estrechos, metas junto
a obstáculos).  Opcionalmente escapa con caminatas aleatorias.
"""
from __future__ import annotations

import time

import numpy as np

from common.geometry import bilinear, segment_free
from common.problem import AlgorithmResult, Problem


def potential_field(problem: Problem, xi=1.0, eta=400.0, rho0_px=25.0, d_star=60.0):
    h, w = problem.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = problem.goal
    d = np.hypot(xx - gx, yy - gy)
    U_att = np.where(d <= d_star, 0.5 * xi * d ** 2, d_star * xi * d - 0.5 * xi * d_star ** 2)
    rho = np.maximum(problem.clearance, 0.5)
    U_rep = np.where(rho < rho0_px, 0.5 * eta * (1.0 / rho - 1.0 / rho0_px) ** 2 * (d_star ** 2), 0.0)
    U = U_att + U_rep
    U[problem.occ] = U.max()
    return U


def run_potential(problem: Problem, rango_repulsion_m=0.6, ganancia_repulsion=1.0, paso_px=1.0,
                  escape_aleatorio=True, intentos_escape=40, semilla=0, max_pasos=20000):
    t0 = time.perf_counter()
    rng = np.random.default_rng(semilla)
    rho0 = max(3.0, rango_repulsion_m / problem.mpp)
    U = potential_field(problem, eta=400.0 * ganancia_repulsion, rho0_px=rho0)
    Gy, Gx = np.gradient(U)
    p = problem.start.astype(float).copy()
    T = problem.goal
    path = [p.copy()]
    best_d = np.hypot(*(p - T))
    stall = 0
    escapes = 0
    minima = []
    for step in range(max_pasos):
        if np.hypot(*(p - T)) <= max(2.0, paso_px * 2) and segment_free(problem.occ, p, T):
            path.append(T.copy())
            break
        g = np.array([bilinear(Gx, p[None])[0], bilinear(Gy, p[None])[0]])
        n = np.hypot(*g)
        if n < 1e-9:
            stall = 10 ** 6
        else:
            q = p - paso_px * g / n
            if segment_free(problem.occ, p, q):
                p = q
                path.append(p.copy())
            else:
                stall += 5
        dcur = np.hypot(*(p - T))
        if dcur < best_d - 0.5:
            best_d = dcur
            stall = 0
        else:
            stall += 1
        if stall > 60:                       # mínimo local
            minima.append(p.copy())
            if not escape_aleatorio or escapes >= intentos_escape:
                break
            escapes += 1
            for _ in range(int(15 + 10 * escapes)):   # caminata aleatoria creciente
                ang = rng.uniform(0, 2 * np.pi)
                q = p + 2.0 * np.array([np.cos(ang), np.sin(ang)])
                if segment_free(problem.occ, p, q):
                    p = q
                    path.append(p.copy())
            best_d = np.hypot(*(p - T))
            stall = 0
    path = np.asarray(path)
    ok = np.hypot(*(path[-1] - T)) < 1e-6
    stats = {"pasos": len(path), "minimos_locales": len(minima), "escapes": escapes}
    viz = {"potential": np.log1p(U - U.min()), "minima": np.asarray(minima) if minima else np.zeros((0, 2)),
           "partial": path}
    if not ok:
        return AlgorithmResult("potential", "Campos potenciales", False, None, time.perf_counter() - t0,
                               "Atrapado en un mínimo local", stats, viz)
    return AlgorithmResult("potential", "Campos potenciales", True, path, time.perf_counter() - t0,
                           "Ruta encontrada" + (f" tras {escapes} escape(s)" if escapes else ""), stats, viz)
