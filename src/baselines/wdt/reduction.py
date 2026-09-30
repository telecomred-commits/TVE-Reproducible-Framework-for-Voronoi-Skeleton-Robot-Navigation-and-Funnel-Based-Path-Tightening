"""Reducción de la ruta (artículo, sección 2.3, figura 6).

1. Se toman η_i y η_j = η_{i+2}.
2. Se traza la recta η_i -> η_j y se verifica colisión sobre el espacio de
   configuración.  Si está libre, se elimina η_{i+1}.  Si no, η_j se mueve al
   punto medio entre η_j y η_{i+1} y se repite; se detiene cuando la distancia
   entre η_j y η_{i+1} es menor que D.  Si nunca queda libre, η_{i+1} es
   irreducible en este ciclo.  Cuando la recta queda libre con un η_j
   intermedio, η_{i+1} se sustituye por ese punto (recorte de esquina).
3. Lo mismo se hace simultáneamente desde el otro extremo (η_i = T,
   η_j = η_{i-2}).
4. El ciclo termina cuando ambos frentes se encuentran en el nodo central.
5. Los ciclos se repiten hasta que todos los nodos son irreducibles.

Cada modificación reduce estrictamente la longitud (desigualdad triangular) y
mantiene la ruta libre de colisión, así que el proceso converge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.geometry import polyline_length, segment_free


@dataclass
class ReductionResult:
    path: np.ndarray                       # ruta reducida final
    history: list[np.ndarray] = field(default_factory=list)  # ruta tras cada ciclo (0 = inicial)
    cycles: int = 0
    removed: int = 0
    moved: int = 0
    refined: bool = False                  # True si se aplicó el tensado (optimización)


def _try_reduce(occ, a, b, c, min_dist, step):
    """Devuelve 'remove', un nuevo punto para b, o None (irreducible)."""
    if segment_free(occ, a, c, step):
        return "remove"
    cand = np.asarray(c, dtype=float)
    b = np.asarray(b, dtype=float)
    while True:
        cand = 0.5 * (cand + b)
        if np.hypot(*(cand - b)) < min_dist:
            return None
        if segment_free(occ, a, cand, step):
            return cand


def reduction_cycle(occ, pts: list, min_dist: float, step: float) -> tuple[list, int, int]:
    W = [np.asarray(p, dtype=float) for p in pts]
    removed = moved = 0
    front = 0
    back = len(W) - 1
    turn = 0
    while back - front >= 2:
        if turn == 0:
            res = _try_reduce(occ, W[front], W[front + 1], W[front + 2], min_dist, step)
            if isinstance(res, str):
                del W[front + 1]
                back -= 1
                removed += 1
            else:
                if res is not None:
                    W[front + 1] = res
                    moved += 1
                front += 1
        else:
            res = _try_reduce(occ, W[back], W[back - 1], W[back - 2], min_dist, step)
            if isinstance(res, str):
                del W[back - 1]
                back -= 1
                removed += 1
            else:
                if res is not None:
                    W[back - 1] = res
                    moved += 1
                back -= 1
        turn ^= 1
    return W, removed, moved


def tighten(occ, path: np.ndarray, sweeps: int = 30, tol: float = 1e-2, iters: int = 12) -> np.ndarray:
    """Optimización adicional (no está en el artículo): tensado de vértices.

    Cada vértice interior V_i se desplaza hacia su proyección sobre la cuerda
    V_{i-1}V_{i+1} lo más posible sin colisión (búsqueda binaria).  La función
    |A-V| + |V-B| es convexa y mínima en la cuerda, por lo que cada paso
    reduce la longitud: el resultado se aproxima a la ruta localmente más
    corta dentro de la misma clase de homotopía.  Los vértices que quedan
    alineados se eliminan.
    """
    P = [np.asarray(p, dtype=float) for p in path]
    for _ in range(sweeps):
        gain = 0.0
        i = 1
        while i < len(P) - 1:
            a, v, b = P[i - 1], P[i], P[i + 1]
            if segment_free(occ, a, b):
                gain += np.hypot(*(v - a)) + np.hypot(*(b - v)) - np.hypot(*(b - a))
                del P[i]
                continue
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else float(np.clip((v - a) @ ab / L2, 0, 1))
            target = a + t * ab
            lo, hi = 0.0, 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                cand = v + mid * (target - v)
                if segment_free(occ, a, cand) and segment_free(occ, cand, b):
                    lo = mid
                else:
                    hi = mid
            if lo > 0:
                nv = v + lo * (target - v)
                gain += (np.hypot(*(v - a)) + np.hypot(*(b - v))) - (np.hypot(*(nv - a)) + np.hypot(*(b - nv)))
                P[i] = nv
            i += 1
        if gain < tol:
            break
    return np.asarray(P)


def reduce_route(occ, waypoints: np.ndarray, min_dist: float, max_cycles: int = 60,
                 step: float = 0.25) -> ReductionResult:
    pts = [np.asarray(p, dtype=float) for p in np.asarray(waypoints, dtype=float)]
    history = [np.asarray(pts)]
    total_removed = total_moved = 0
    cycles = 0
    min_dist = max(float(min_dist), 0.5)
    for _ in range(max_cycles):
        before = polyline_length(np.asarray(pts))
        new_pts, removed, moved = reduction_cycle(occ, pts, min_dist, step)
        cycles += 1
        after = polyline_length(np.asarray(new_pts))
        pts = new_pts
        history.append(np.asarray(pts))
        total_removed += removed
        total_moved += moved
        if removed == 0 and (moved == 0 or before - after < 1e-3):
            break
    return ReductionResult(np.asarray(pts), history, cycles, total_removed, total_moved)
