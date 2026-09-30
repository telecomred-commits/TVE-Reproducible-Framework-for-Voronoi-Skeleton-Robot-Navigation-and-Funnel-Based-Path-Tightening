"""Planificación topológica: clases de homotopía y homología.

Idea
----
1. **Obstáculos como variedades.**  Cada obstáculo del espacio de configuración
   (componente conexa que no toca el borde del mapa) es una 2-variedad
   compacta con borde.  Su característica de Euler χ = 1 − (nº de agujeros) la
   clasifica: si χ = 1 es homeomorfa al disco cerrado D² (su frontera es una
   1-esfera S¹).  Un disco es contráctil, así que se puede contraer a un punto
   ζᵢ (retracto por deformación) sin cambiar la topología del espacio libre.

2. **Espacio libre.**  Queda homotópicamente equivalente a un disco con n
   puntos removidos, cuyo grupo fundamental es el grupo libre Fₙ = ⟨x₁,…,xₙ⟩
   (clases de HOMOTOPÍA de caminos S→T) y cuya primera homología es
   H₁ = ℤⁿ (clases de HOMOLOGÍA).

3. **Invariantes calculables.**  Desde cada ζᵢ se traza un rayo vertical hacia
   el borde superior.  Al recorrer un camino se anota xᵢ al cruzar el rayo i de
   izquierda a derecha y xᵢ⁻¹ al revés; la PALABRA REDUCIDA (xᵢ xᵢ⁻¹ = 1) es un
   invariante completo de homotopía.  Su abelianización — el vector de cruces
   netos hᵢ ∈ ℤ — es el invariante de homología: dos caminos pueden ser
   homólogos sin ser homotópicos (p. ej. x₁x₂x₁⁻¹x₂⁻¹ ≠ 1 pero su vector es 0).

4. **Búsqueda.**  A* sobre el grafo aumentado (celda, palabra): cada vez que se
   extrae T con una palabra nueva se obtiene la ruta más corta (en la malla)
   de esa clase de homotopía, en orden creciente de longitud.

5. **Geodésica de cada clase.**  Cada ruta se tensa moviendo sus vértices sólo
   si los triángulos barridos no contienen obstáculos (homotopía por rectas),
   así se obtiene el camino más corto de la clase sin salir de ella.  La ruta
   elegida es la más corta entre todas las clases encontradas.
"""
from __future__ import annotations

import heapq
import math
import time

import cv2
import numpy as np

from common.geometry import polyline_length, segment_free
from common.problem import AlgorithmResult, Problem, cell_center, coarse_grid
from baselines.grid_search import _attach

SQ2 = math.sqrt(2.0)
SUB = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉⁻")


# --------------------------------------------------------------------------- #
# 1. Análisis topológico de los obstáculos
# --------------------------------------------------------------------------- #
def analyze_obstacles(occ: np.ndarray, min_area: int = 4) -> list[dict]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
    h, w = occ.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue                         # unido al borde: es parte de la frontera exterior
        if area < min_area:
            continue
        mask = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        pad = np.pad(mask, 1)
        cnts, hier = cv2.findContours(pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = int(sum(1 for k in range(len(cnts)) if hier[0][k][3] >= 0))
        chi = 1 - holes
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)
        py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
        if holes == 0:
            kind = "disco D² (frontera ≅ S¹) — contráctil a un punto"
        elif holes == 1:
            kind = "anillo S¹×[0,1] (χ = 0)"
        else:
            kind = f"disco con {holes} agujeros (χ = {chi})"
        out.append({"id": len(out) + 1, "label": i, "area_px": int(area), "holes": holes, "euler": chi,
                    "type": kind, "point": np.array([x + px - 1.0, y + py - 1.0]),
                    "bbox": (int(x), int(y), int(bw), int(bh))})
    return out


def word_str(word) -> str:
    if not word:
        return "1 (trivial)"
    parts = []
    for l in word:
        s = f"x{str(abs(l)).translate(SUB)}"
        if l < 0:
            s += "⁻¹"
        parts.append(s)
    return "·".join(parts)


def _reduce_append(word: tuple, letter: int) -> tuple:
    if word and word[-1] == -letter:
        return word[:-1]
    return word + (letter,)


def polyline_word(path: np.ndarray, rays: list[tuple[int, float, float]]) -> tuple:
    """Palabra de cruces de una poligonal continua con los rayos (x = rx, y < zy)."""
    word: tuple = ()
    for a, b in zip(path[:-1], path[1:]):
        events = []
        for letter, rx, zy in rays:
            if (a[0] - rx) * (b[0] - rx) < 0:
                t = (rx - a[0]) / (b[0] - a[0])
                y = a[1] + t * (b[1] - a[1])
                if y < zy:
                    events.append((t, letter if b[0] > a[0] else -letter))
        for _, l in sorted(events):
            word = _reduce_append(word, l)
    return word


def homology(word, n: int) -> tuple:
    v = [0] * n
    for l in word:
        v[abs(l) - 1] += 1 if l > 0 else -1
    return tuple(v)


# --------------------------------------------------------------------------- #
# 5. Tensado que preserva la clase de homotopía
# --------------------------------------------------------------------------- #
def _triangle_empty(occ, a, b, c) -> bool:
    pts = np.array([a, b, c], float)
    x0 = int(max(np.floor(pts[:, 0].min()) - 1, 0))
    y0 = int(max(np.floor(pts[:, 1].min()) - 1, 0))
    x1 = int(min(np.ceil(pts[:, 0].max()) + 2, occ.shape[1]))
    y1 = int(min(np.ceil(pts[:, 1].max()) + 2, occ.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return True
    sub = occ[y0:y1, x0:x1]
    if not sub.any():
        return True
    m = np.zeros(sub.shape, np.uint8)
    poly = np.round((pts - [x0, y0]) * 4).astype(np.int32)
    cv2.fillPoly(m, [poly], 1, shift=2)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))              # tolera el contacto con el borde
    return not (sub & (m > 0)).any()


def tighten_homotopic(occ, path: np.ndarray, sweeps: int = 40, iters: int = 10) -> np.ndarray:
    P = [np.asarray(p, float) for p in path]
    for _ in range(sweeps):
        gain = 0.0
        i = 1
        while i < len(P) - 1:
            a, v, b = P[i - 1], P[i], P[i + 1]
            if segment_free(occ, a, b) and _triangle_empty(occ, a, v, b):
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
                c = v + mid * (target - v)
                if (segment_free(occ, a, c) and segment_free(occ, c, b)
                        and _triangle_empty(occ, a, v, c) and _triangle_empty(occ, v, c, b)):
                    lo = mid
                else:
                    hi = mid
            if lo > 0:
                nv = v + lo * (target - v)
                gain += (np.hypot(*(v - a)) + np.hypot(*(b - v))) - (np.hypot(*(nv - a)) + np.hypot(*(b - nv)))
                P[i] = nv
            i += 1
        if gain < 1e-2:
            break
    return np.asarray(P)


# --------------------------------------------------------------------------- #
# 4. A* en el grafo aumentado (celda, palabra)
# --------------------------------------------------------------------------- #
def run_topological(problem: Problem, celda=3, clases=6, longitud_palabra=4, max_expansiones=600000):
    t0 = time.perf_counter()
    cell = max(1, int(celda))
    obstacles = analyze_obstacles(problem.occ, min_area=max(4, cell * cell))
    n_obs = len(obstacles)
    grid = coarse_grid(problem.occ, cell)
    H, W = grid.shape
    s = _attach(problem, grid, cell, problem.start)
    g = _attach(problem, grid, cell, problem.goal)
    if s is None or g is None:
        return AlgorithmResult("topological", "Topológico", False, None, time.perf_counter() - t0,
                               "S o T no se pueden conectar a la malla")
    # rayos en coordenadas de celda (centro de celda = entero) y en píxeles
    gap_rays: dict[int, list] = {}
    rays_px = []
    for k, ob in enumerate(obstacles):
        zx = (ob["point"][0] - (cell - 1) / 2.0) / cell
        zy = (ob["point"][1] - (cell - 1) / 2.0) / cell
        rx = math.floor(zx) + 0.5 + 1e-3 * (k + 1)           # entre columnas, nunca sobre un centro
        gap = math.floor(rx)
        gap_rays.setdefault(gap, []).append((k + 1, rx, zy))
        rays_px.append((k + 1, rx * cell + (cell - 1) / 2.0, zy * cell + (cell - 1) / 2.0))
        ob["ray_x_px"] = rays_px[-1][1]
    Wp = W + 2
    blocked = np.ones((H + 2, W + 2), bool)
    blocked[1:-1, 1:-1] = grid
    blk = blocked.ravel().tolist()
    si = (s[0] + 1) * Wp + s[1] + 1
    gi = (g[0] + 1) * Wp + g[1] + 1
    gr, gc = g[0] + 1, g[1] + 1
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]

    def heur(i):
        r, c = divmod(i, Wp)
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (SQ2 - 2) * min(dr, dc)

    start_state = (si, ())
    gbest = {start_state: 0.0}
    parent = {start_state: None}
    heap = [(heur(si), 0.0, 0, start_state)]
    closed = set()
    counter = 1
    found = []
    words_found = set()
    expanded = 0
    maxlen = int(longitud_palabra)
    while heap and len(found) < int(clases) and expanded < max_expansiones:
        f, gcost, _, state = heapq.heappop(heap)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1
        u, word = state
        if u == gi:
            if word not in words_found:
                words_found.add(word)
                found.append((gcost, state))
            continue
        ur, uc = divmod(u, Wp)
        for dr, dc, cost in moves:
            v = u + dr * Wp + dc
            if blk[v]:
                continue
            if dr and dc and (blk[u + dr * Wp] or blk[u + dc]):
                continue
            nw = word
            if dc:
                gap = (uc - 1) if dc > 0 else (uc - 2)       # columnas en coordenadas sin borde
                rays = gap_rays.get(gap)
                if rays:
                    r1 = ur - 1
                    for letter, rx, zy in sorted(rays, key=lambda t: t[1], reverse=dc < 0):
                        c1 = uc - 1
                        t = (rx - c1) / dc
                        y = r1 + dr * t
                        if y < zy:
                            nw = _reduce_append(nw, letter if dc > 0 else -letter)
            if len(nw) > maxlen:
                continue
            ns = (v, nw)
            if ns in closed:
                continue
            ng = gcost + cost
            if ng < gbest.get(ns, math.inf):
                gbest[ns] = ng
                parent[ns] = state
                counter += 1
                heapq.heappush(heap, (ng + heur(v), ng, counter, ns))

    classes = []
    for gcost, state in found:
        chain = [state]
        while parent[chain[-1]] is not None:
            chain.append(parent[chain[-1]])
        chain.reverse()
        pts = [problem.start] + [cell_center(i // Wp - 1, i % Wp - 1, cell) for i, _ in chain] + [problem.goal]
        grid_path = np.asarray(pts, float)
        word = state[1]
        taut = tighten_homotopic(problem.occ, grid_path)
        ok_class = polyline_word(taut, rays_px) == word
        if not ok_class:                     # salvaguarda: nunca cambiar de clase
            taut = grid_path
        classes.append({"word": word, "word_str": word_str(word), "homology": homology(word, n_obs),
                        "grid_path": grid_path, "path": taut, "grid_length_px": polyline_length(grid_path),
                        "length_px": polyline_length(taut), "class_preserved": ok_class})
    # grupos de homología (clases de homotopía distintas con el mismo vector)
    hom_groups: dict[tuple, list[int]] = {}
    for k, c in enumerate(classes):
        hom_groups.setdefault(c["homology"], []).append(k)
    for c in classes:
        c["homologous_with"] = [k for k in hom_groups[c["homology"]]]
    stats = {"obstaculos_puntuales": n_obs, "clases_homotopia": len(classes),
             "clases_homologia": len(hom_groups), "estados_expandidos": expanded, "celda_px": cell,
             "rango_H1": n_obs, "euler_espacio_libre": 1 - n_obs}
    viz = {"obstacles": obstacles, "rays": rays_px, "classes": classes}
    if not classes:
        return AlgorithmResult("topological", "Topológico", False, None, time.perf_counter() - t0,
                               "No se encontró ninguna clase de homotopía S→T", stats, viz)
    best = int(np.argmin([c["length_px"] for c in classes]))
    viz["best"] = best
    stats["clase_elegida"] = classes[best]["word_str"]
    return AlgorithmResult("topological", "Topológico", True, classes[best]["path"], time.perf_counter() - t0,
                           f"{len(classes)} clases de homotopía ({len(hom_groups)} de homología); "
                           f"la más corta es {classes[best]['word_str']}", stats, viz)
