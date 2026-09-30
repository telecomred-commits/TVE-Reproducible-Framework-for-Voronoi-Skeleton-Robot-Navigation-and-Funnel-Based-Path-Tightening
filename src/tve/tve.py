"""TVE — Teselado Voronoi + Embudo: versión mejorada del método del artículo.

Mantiene la idea del artículo (teselar el espacio libre desde los obstáculos,
navegar por el esqueleto y reducir la ruta usando las líneas transversales),
pero reemplaza cada etapa por un algoritmo con mejor complejidad o garantías:

1. **Teselado en una pasada.**  El teselado por frentes de onda es un diagrama
   de Voronoi discreto.  Se calcula con ``cv2.distanceTransformWithLabels``
   (distancia euclidiana, algoritmo lineal en C) en vez de expandir frentes
   iteración a iteración.  Da además, para cada píxel, su obstáculo más cercano.
2. **Caché multi-consulta.**  Teselado y esqueleto no dependen de S y T: se
   calculan una vez por mapa y robot; cada consulta nueva sólo conecta S y T.
3. **Rutas con garantías.**  Dijkstra sobre el grafo del esqueleto (ruta más
   corta del grafo) + K alternativas por penalización de aristas (rodean
   obstáculos distintos), en lugar de la ruta voraz + n−1 alternativas.
4. **Algoritmo del embudo.**  Cada punto del esqueleto equidista de sus dos
   obstáculos generadores; el segmento entre los dos puntos de obstáculo más
   cercanos es un *portal* (la «línea transversal» del artículo) totalmente
   libre.  La ruta más corta que cruza los portales en orden se obtiene con el
   algoritmo del embudo (Lee y Preparata 1984; versión de Mononen) en O(n), sin
   la reducción iterativa ni el parámetro D.  Alternativa robusta: portales
   como diámetros del disco libre de cada punto del esqueleto (el esqueleto
   los cruza todos, así que el embudo nunca es más largo que él).  Por
   defecto se evalúan ambos tipos y se conserva la ruta más corta.
5. **Chequeos de colisión baratos**: sin chequeos durante la búsqueda; en la
   validación y la poda, aceptación rápida con la transformada de distancia
   exacta (cubrimiento por bolas libres) y prueba exacta sólo cerca de
   obstáculos.  Las candidatas cuya cota inferior no mejora la mejor ruta se
   descartan sin ejecutar el embudo.  S y T se conectan por su radio de
   Voronoi (sin BFS).
   Generadores automáticos: si el esqueleto no cubre una región libre (p.ej.
   laberinto de un solo obstáculo) se dividen los contornos en tramos.
6. **Replanificación dinámica**: con el teselado en pocos ms, recalcular tras
   un cambio del mapa es más barato que la actualización del artículo.
7. **Suavizado con curvatura acotada**: filetes circulares de radio R_min en
   cada esquina (radio adaptativo si no cabe), dentro de un corredor con
   holgura extra, para vehículos tipo Ackermann.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from common.cspace import configuration_space, robot_radius_px
from common.environment import Environment
from common.geometry import FastChecker, densify, points_in_collision, polyline_free, polyline_length, simplify_collision_free
from common.skeleton import (SkeletonGraph, attach_point, bridge_components, extract_skeleton, free_components,
                       pixel_of)
from common.smoothing import smooth_piecewise
from common.tessellation import Tessellation, generator_labels


# --------------------------------------------------------------------------- #
# 1. Teselado de Voronoi euclidiano en una pasada
# --------------------------------------------------------------------------- #
def contour_generators(occ: np.ndarray, segment_px: float):
    """Generadores por LONGITUD DE ARCO: cada contorno (exterior o de hueco) de
    cada obstáculo se divide en tramos de ~``segment_px`` a lo largo del
    contorno.  A diferencia de dividir con una cuadrícula fija, las dos caras
    opuestas de un pasillo formado por un mismo obstáculo quedan en posiciones
    de arco distintas y, por lo tanto, en generadores distintos: aparece la
    frontera central del pasillo.  Devuelve (etiquetas, nº, comp_of_label)."""
    occ_u8 = occ.astype(np.uint8)
    _, comp = cv2.connectedComponents(occ_u8, connectivity=8)
    contours, _ = cv2.findContours(occ_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lab = np.zeros(occ.shape, np.int32)
    comp_of = [0]
    nxt = 1
    s = max(float(segment_px), 4.0)
    for c in contours:
        pts = c[:, 0, :]
        if len(pts) == 0:
            continue
        seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        perim = float(seg.sum())
        nseg = max(1, int(round(perim / s)))
        k = np.minimum((cum / max(perim, 1e-9) * nseg).astype(int), nseg - 1)
        lab[pts[:, 1], pts[:, 0]] = nxt + k
        comp_of.extend([int(comp[pts[0, 1], pts[0, 0]])] * nseg)
        nxt += nseg
    # el resto de píxeles ocupados hereda la etiqueta del contorno más cercano
    src = (lab == 0).astype(np.uint8)
    if src.all():
        return comp.astype(np.int32), int(comp.max()), np.arange(int(comp.max()) + 1, dtype=np.int32)
    _, nl = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(src == 0)
    lut = np.zeros(int(nl.max()) + 1, np.int32)
    lut[nl[zy, zx]] = lab[zy, zx]
    gen = np.where(occ, lut[nl], 0).astype(np.int32)
    return gen, nxt - 1, np.asarray(comp_of, np.int32)


def voronoi_tessellation(occ: np.ndarray, mode: str = "obstacles", segment_px: float = 0.0):
    """Devuelve (Tessellation, nearest) con ``nearest[r, c] = (x, y)`` del píxel
    de obstáculo más cercano a cada píxel."""
    occ = np.asarray(occ, dtype=bool)
    if mode == "contours":
        gen, n_tiles, comp_of = contour_generators(occ, segment_px)
    else:
        gen, n_tiles, comp_of, _ = generator_labels(occ, mode, segment_px)
    free_u8 = (~occ).astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(free_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(occ)
    lz = lab[zy, zx]
    size = int(lab.max()) + 1
    t_gen = np.zeros(size, np.int32)
    t_x = np.zeros(size, np.float32)
    t_y = np.zeros(size, np.float32)
    t_gen[lz] = gen[zy, zx]
    t_x[lz] = zx
    t_y[lz] = zy
    labels = t_gen[lab]
    nearest = np.stack([t_x[lab], t_y[lab]], axis=-1)
    tess = Tessellation(labels=labels, n_tiles=int(n_tiles), wave=np.rint(dist).astype(np.int32),
                        collision=np.zeros(occ.shape, bool), clearance=dist.astype(np.float32), occ=occ.copy(),
                        iterations=0, comp_of_label=comp_of, mode=mode)
    return tess, nearest


def _free(occ, pts) -> bool:
    """Prueba de colisión exacta; con un :class:`FastChecker` usa primero la
    aceptación rápida por transformada de distancia (mismo resultado)."""
    if isinstance(occ, FastChecker):
        return occ.free(pts)
    return polyline_free(occ, pts)


# --------------------------------------------------------------------------- #
# 4. Algoritmo del embudo
# --------------------------------------------------------------------------- #
def _cross(o, a, b):
    """z de (a − o) × (b − o).  En coordenadas de imagen (y hacia abajo),
    un valor negativo significa que b está a la IZQUIERDA de o→a."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def funnel(portals) -> tuple[np.ndarray, list[int]]:
    """Camino más corto que cruza en orden una secuencia de portales (izq., der.).

    El primer y el último portal deben ser degenerados (S y T).  Implementación
    del «simple stupid funnel» de Mononen con números nativos de Python.
    Devuelve (puntos, índice de portal de cada punto).
    """
    # Productos cruzados en línea (sin llamadas) sobre listas de floats nativos:
    # _cross(o, a, b) = (ax − ox)(by − oy) − (ay − oy)(bx − ox)
    A = np.asarray([[a[0], a[1], b[0], b[1]] for a, b in portals], float).tolist()
    n = len(A)
    ax, ay = A[0][0], A[0][1]                     # ápice
    lx, ly = ax, ay                               # borde izquierdo
    rx, ry = A[0][2], A[0][3]                     # borde derecho
    ai = li = ri = 0
    path = [(ax, ay)]
    idx = [0]
    i = 1
    E = 1e-9
    while i < n:
        plx, ply, prx, pry = A[i]
        # lado derecho: el embudo se estrecha si el nuevo punto derecho queda a
        # la izquierda (o sobre) del borde derecho actual
        if (rx - ax) * (pry - ay) - (ry - ay) * (prx - ax) <= 0:
            if (abs(ax - rx) < E and abs(ay - ry) < E) or \
                    (lx - ax) * (pry - ay) - (ly - ay) * (prx - ax) > 0:
                rx, ry, ri = prx, pry, i
            else:                                   # cruzó el borde izquierdo: nuevo vértice
                ax, ay, ai = lx, ly, li
                path.append((ax, ay))
                idx.append(ai)
                rx, ry = lx, ly
                ri = li = ai
                i = ai + 1
                continue
        # lado izquierdo (simétrico)
        if (lx - ax) * (ply - ay) - (ly - ay) * (plx - ax) >= 0:
            if (abs(ax - lx) < E and abs(ay - ly) < E) or \
                    (rx - ax) * (ply - ay) - (ry - ay) * (plx - ax) < 0:
                lx, ly, li = plx, ply, i
            else:
                ax, ay, ai = rx, ry, ri
                path.append((ax, ay))
                idx.append(ai)
                lx, ly = rx, ry
                ri = li = ai
                i = ai + 1
                continue
        i += 1
    ex, ey = A[-1][0], A[-1][1]
    if not (abs(path[-1][0] - ex) < E and abs(path[-1][1] - ey) < E):
        path.append((ex, ey))
        idx.append(n - 1)
    return np.asarray(path, float), idx


def portal_lower_bound(portals, S, T) -> float:
    """Cota inferior de la longitud de cualquier ruta S→T que cruce todos los
    portales: max_i [d(S, portal_i) + d(portal_i, T)].  Permite descartar
    candidatas sin ejecutar el embudo."""
    Pt = np.asarray(portals, float)
    A, B = Pt[:, 0], Pt[:, 1]
    d = B - A
    L2 = np.maximum((d * d).sum(1), 1e-12)

    def dist(q):
        t = np.clip(((q - A) * d).sum(1) / L2, 0, 1)
        X = A + t[:, None] * d
        return np.hypot(X[:, 0] - q[0], X[:, 1] - q[1])

    return float((dist(np.asarray(S, float)) + dist(np.asarray(T, float))).max())


def prune_vertices(occ, path: np.ndarray, shortcuts: bool = True) -> np.ndarray:
    """Quita los vértices sobrantes (escalones de la cuantización al píxel).

    1) Douglas-Peucker (0.75 px, en C, sin chequeos) elimina el jitter casi
       colineal; se acepta sólo si la ruta simplificada sigue libre.
    2) Opcional: atajos con chequeo de colisión (el embudo ya es tenso, así que
       normalmente no hacen falta)."""
    path = np.asarray(path, float)
    if len(path) > 3:
        dp = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), 0.75, False).reshape(-1, 2)
        dp = dp.astype(float)
        dp[0], dp[-1] = path[0], path[-1]
        # Si alguna cuerda de DP roza un obstáculo, simplificación recursiva
        # guiada por colisión (≈ 2 chequeos por vértice resultante).
        path = dp if _free(occ, dp) else simplify_collision_free(occ, path)
    if not shortcuts:
        return path
    # Con un FastChecker los atajos se aceptan sólo si pasan la prueba rápida
    # (conservadora): así no rozan obstáculos y conservan la holgura del embudo.
    ok = occ.accepts if isinstance(occ, FastChecker) else (lambda q: polyline_free(occ, q))
    P = [np.asarray(p, float) for p in path]
    i = 1
    while i < len(P) - 1:
        if ok(np.vstack([P[i - 1], P[i + 1]])):
            del P[i]
            i = max(1, i - 1)
        else:
            i += 1
    return np.asarray(P)


def skeleton_defects(g: SkeletonGraph, occ: np.ndarray, free_comp: np.ndarray, min_area: int = 50,
                     min_frac: float = 0.02) -> int:
    """Nº de defectos de conectividad del esqueleto: por cada región libre
    navegable (área ≥ ``min_area`` px y ≥ ``min_frac`` de la mayor), |nº de
    componentes del esqueleto dentro − 1|.  0 = cada región libre tiene
    exactamente un esqueleto conexo.  Los bolsillos pequeños (p. ej. entre una
    silla y un escritorio en un mapa percibido) no cuentan."""
    area = np.bincount(free_comp.ravel())
    area[0] = 0
    thr = max(min_area, min_frac * float(area.max()))
    big = [int(k) for k in np.flatnonzero(area >= thr) if k != 0]
    seen = {k: set() for k in big}
    comp = g.components()
    for nid, xy in g.nodes.items():
        pix = pixel_of(occ, xy)
        if pix is None:
            continue
        fc = int(free_comp[pix])
        if fc in seen:
            seen[fc].add(comp[nid])
    return int(sum(abs(len(v) - 1) for v in seen.values()))


def build_corner_index(g: SkeletonGraph):
    """Índice (precalculado una vez por mapa) de todos los puntos del esqueleto."""
    coords, eids, ks = [], [], []
    for eid, e in g.edges.items():
        if e.kind != "skeleton":
            continue
        n = len(e.pts)
        coords.append(e.pts)
        eids.append(np.full(n, eid))
        ks.append(np.arange(n))
    if not coords:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int)
    return np.vstack(coords), np.concatenate(eids), np.concatenate(ks)


def find_attach(prep, p_xy):
    """Punto del esqueleto al que conectar ``p``: primero por su RADIO DE
    VORONOI (desde el obstáculo más cercano, a través de p, hasta la frontera
    entre teselas) para que p quede sobre el portal de su nodo y la ruta no
    retroceda; si no, el punto visible más cercano.

    Si ``p`` está en una grieta (sin visibilidad directa) se sube primero por
    el gradiente de la holgura hasta salir de ella.
    Devuelve (eid, k, via) —via: puntos intermedios del conector— o None."""
    if len(prep.corners[0]) == 0:
        return None
    p = np.asarray(p_xy, float)
    ref = _attach_direct(prep, p, n_near=16)
    if ref is not None:
        return ref[0], ref[1], []
    via = _climb(prep, p)
    if len(via) and _free(prep.checker, np.vstack([p, via])):
        ref = _attach_direct(prep, via[-1], n_near=16)
        if ref is not None:
            return ref[0], ref[1], [q for q in simplify_collision_free(prep.checker, np.vstack([p, via]))[1:]]
    return None


def _climb(prep, p, max_steps: int = 30) -> np.ndarray:
    """Ascenso por la transformada de distancia (vecindad 8) desde p."""
    dt = prep.checker.dt
    h, w = dt.shape
    r, c = int(round(p[1])), int(round(p[0]))
    out = []
    for _ in range(max_steps):
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, h), max(c - 1, 0), min(c + 2, w)
        win = dt[r0:r1, c0:c1]
        k = int(np.argmax(win))
        rr, cc = r0 + k // win.shape[1], c0 + k % win.shape[1]
        if win.flat[k] <= dt[r, c]:
            break
        r, c = rr, cc
        out.append((float(c), float(r)))
    return np.asarray(out, float).reshape(-1, 2)


def _attach_direct(prep, p, n_near: int = 16):
    tess, nearest = prep.tess, prep.nearest
    coords, eids, ks = prep.corners
    h, w = tess.labels.shape
    r0, c0 = int(round(p[1])), int(round(p[0]))
    dd = None
    a = nearest[r0, c0].astype(float)
    d = p - a
    dn = float(np.hypot(*d))
    if dn > 1e-6:
        d /= dn
        lab0 = tess.labels[r0, c0]
        s = np.arange(0.5, prep.spoke_max, 0.5)
        Q = p[None, :] + s[:, None] * d[None, :]
        R = np.round(Q[:, 1]).astype(int)
        C = np.round(Q[:, 0]).astype(int)
        inside = (R >= 0) & (R < h) & (C >= 0) & (C < w)
        stop = np.flatnonzero(~inside)
        lim = stop[0] if len(stop) else len(s)
        R, C = R[:lim], C[:lim]
        blocked = tess.occ[R, C]
        change = tess.labels[R, C] != lab0
        first_block = np.flatnonzero(blocked)
        first_change = np.flatnonzero(change)
        if len(first_change) and (not len(first_block) or first_change[0] < first_block[0]):
            hit = Q[first_change[0]]
            dd = np.hypot(coords[:, 0] - hit[0], coords[:, 1] - hit[1])
            k = _first_visible(prep.checker, p, coords, np.argsort(dd)[:8])
            if k is not None:
                return int(eids[k]), int(ks[k])
    dx, dy = coords[:, 0] - p[0], coords[:, 1] - p[1]
    dd = np.hypot(dx, dy)
    order = np.argsort(dd)
    # Candidatos: los más cercanos y, para cubrir todas las direcciones con
    # pocos chequeos, el más cercano de cada uno de 16 sectores angulares.
    sec = ((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * 16).astype(int) % 16
    first = {}
    for j in order:
        s = int(sec[j])
        if s not in first:
            first[s] = int(j)
            if len(first) == 16:
                break
    cand = list(dict.fromkeys([int(j) for j in order[:n_near // 2]] +
                              sorted(first.values(), key=lambda j: dd[j])))
    k = _first_visible(prep.checker, p, coords, cand)
    if k is not None:
        return int(eids[k]), int(ks[k])
    return None


def _first_visible(chk: FastChecker, p, coords, cand):
    """Primer candidato visible desde p: primero con la prueba rápida en todos
    (barata) y sólo después con la prueba exacta."""
    for k in cand:
        if chk.accepts(np.vstack([p, coords[k]])):
            return k
    for k in cand:
        if polyline_free(chk.occ, np.vstack([p, coords[k]])):
            return k
    return None


def find_attach_multi(prep, p_xy, sectors: int = 8, per_sector: int = 2, max_checks: int = 16):
    """Varios puntos de conexión para ``p``: el del radio de Voronoi
    (:func:`find_attach`) y, en cada uno de ``sectors`` sectores angulares, el
    punto visible más cercano del esqueleto.  Con varios conectores Dijkstra
    elige el que sale hacia el objetivo y la ruta no retrocede.
    Devuelve una lista de (eid, k, via) (vacía si no hay ninguno)."""
    base = find_attach(prep, p_xy)
    if base is None:
        return []
    refs = [base]
    if base[2]:                    # p en una grieta: un único conector con escalada
        return refs
    coords, eids, ks = prep.corners
    p = np.asarray(p_xy, float)
    dx, dy = coords[:, 0] - p[0], coords[:, 1] - p[1]
    dd = np.hypot(dx, dy)
    dmin = float(dd.min())
    near = np.flatnonzero(dd <= 4.0 * dmin + 20.0)
    near = near[np.argsort(dd[near])]
    sec = ((np.arctan2(dy[near], dx[near]) + math.pi) / (2 * math.pi) * sectors).astype(int) % sectors
    tried = {}
    checks = 0
    for j, s in zip(near, sec):
        if checks >= max_checks or len(tried) == sectors and all(v < 0 or v >= per_sector
                                                                 for v in tried.values()):
            break
        n_s = tried.get(s, 0)
        if n_s < 0 or n_s >= per_sector:           # sector resuelto o agotado
            continue
        checks += 1
        # conectores adicionales: sólo la prueba rápida (conservadora, con holgura)
        if prep.checker.accepts(np.vstack([p, coords[j]])):
            refs.append((int(eids[j]), int(ks[j]), []))
            tried[s] = -1
        else:
            tried[s] = n_s + 1
    return list(dict.fromkeys((r[0], r[1], tuple(map(tuple, r[2]))) for r in refs).keys())


def attach_multi(g: SkeletonGraph, items):
    """Conecta varios puntos, cada uno con varios conectores.

    ``items``: lista de (p, kind, refs) con refs = [(eid, k, via), ...].
    Las divisiones de una misma arista se hacen de mayor a menor ``k``: así la
    primera mitad (u → nuevo) conserva los índices de las siguientes.
    Devuelve los id de nodo de cada punto."""
    by_edge: dict[int, set] = {}
    for _, _, refs in items:
        for eid, k, _ in refs:
            by_edge.setdefault(eid, set()).add(k)
    target: dict[tuple[int, int], int] = {}
    for eid, kset in by_edge.items():
        cur = eid
        for k in sorted(kset, reverse=True):
            e = g.edges[cur]
            n = len(e.pts)
            if k <= 0:
                target[(eid, k)] = e.u
                continue
            if k >= n - 1:
                target[(eid, k)] = e.v
                continue
            nid = g.add_node(e.pts[k], "split")
            g.remove_edge(cur)
            first = g.add_edge(e.u, nid, e.pts[:k + 1], e.tiles, e.kind)
            g.add_edge(nid, e.v, e.pts[k:], e.tiles, e.kind)
            target[(eid, k)] = nid
            cur = first
    out = []
    for p, kind, refs in items:
        p = np.asarray(p, float).reshape(1, 2)
        nid = g.add_node(p[0], kind)
        done = set()
        for eid, k, via in refs:
            t = target[(eid, k)]
            if t in done:
                continue
            done.add(t)
            pts = np.vstack([p] + [np.reshape(q, (1, 2)) for q in via] + [np.reshape(g.nodes[t], (1, 2))])
            g.add_edge(nid, t, pts, kind="connector")
        out.append(nid)
    return out


def attach_pair(g: SkeletonGraph, S, T, ref_s, ref_t):
    """Conecta S y T a sus puntos del esqueleto (dividiendo aristas).  Si ambos
    caen en la misma arista se ajustan los índices tras la primera división."""
    def conn(p, via, target):
        return np.vstack([np.asarray(p, float).reshape(1, 2)] + [np.reshape(q, (1, 2)) for q in via]
                         + [np.reshape(g.nodes[target], (1, 2))])

    def split(eid, k, p, via, kind):
        e = g.edges[eid]
        n = len(e.pts)
        u, v = e.u, e.v
        target = g.split_edge(eid, k, "split") if 0 < k < n - 1 else (u if k <= 0 else v)
        nid = g.add_node(p, kind)
        g.add_edge(nid, target, conn(p, via, target), kind="connector")
        return nid, target
    ref_s = tuple(ref_s) + ((),) * (3 - len(ref_s))
    ref_t = tuple(ref_t) + ((),) * (3 - len(ref_t))
    s_node, s_target = split(ref_s[0], ref_s[1], S, ref_s[2], "start")
    eid, k, via_t = ref_t
    if eid == ref_s[0] and eid not in g.edges:
        ks = ref_s[1]
        if k == ks:
            nid = g.add_node(T, "goal")
            g.add_edge(nid, s_target, conn(T, via_t, s_target), kind="connector")
            return s_node, nid
        # la arista original se dividió en (u → s_target) y (s_target → v)
        for e2 in g.adj[s_target]:
            ed = g.edges[e2]
            if ed.kind != "skeleton":
                continue
            if k < ks and ed.v == s_target:
                eid, k = e2, k
                break
            if k > ks and ed.u == s_target:
                eid, k = e2, k - ks
                break
    t_node, _ = split(eid, k, T, via_t, "goal")
    return s_node, t_node


def disc_portals(route_pts: np.ndarray, tess: Tessellation, nearest: np.ndarray, dt: np.ndarray,
                 shrink_px: float, stride: int = 3, min_cross: float = 30.0, win: int = 7):
    """Portales como DIÁMETROS DEL DISCO LIBRE de cada punto de la ruta.

    Para cada punto p de la ruta del esqueleto:
      * radio libre exacto ρ = dt(c) − |p − c| − √2/2 (c = píxel de p; ver
        :class:`FastChecker`), menos el margen ``shrink_px``;
      * dirección u = bisectriz hacia los obstáculos generadores (a y b son
        los obstáculos más cercanos de las dos teselas vecinas):
        u ∝ (b − p)/|b − p| − (a − p)/|a − p|; si no hay dos teselas o u queda
        casi paralela a la ruta, se usa la normal a la ruta;
      * portal = [p − ρ·u, p + ρ·u] (izquierda, derecha).
    Así cada portal es libre (está dentro del disco) y la propia ruta del
    esqueleto lo cruza en p: el embudo nunca es más largo que el esqueleto.
    Devuelve (portales, índices en la ruta).
    """
    h, w = dt.shape
    sel = _subsample(len(route_pts), stride)          # los puntos del esqueleto están a ~1 px
    R = np.asarray(route_pts, float)[sel]
    n = len(R)
    stride_eff = max(1, stride)
    win = max(3, (win // stride_eff) | 1)
    # tangente suavizada (≈ 8 px de base)
    m = max(1, int(round(4 / stride_eff)))
    tg = np.zeros_like(R)
    if n > 2 * m:
        tg[m:-m] = R[2 * m:] - R[:-2 * m]
        tg[:m] = tg[m]
        tg[-m:] = tg[-m - 1]
    else:
        tg[:] = R[-1] - R[0]
    tn = np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-9)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1) / tn[:, None]          # normal (lado derecho)
    # obstáculos de las teselas vecinas (puntos del esqueleto = esquinas)
    fx, fy = R[:, 0] + 0.5, R[:, 1] + 0.5
    corner = (np.abs(fx - np.round(fx)) < 1e-6) & (np.abs(fy - np.round(fy)) < 1e-6)
    j = np.round(fx).astype(int)
    i = np.round(fy).astype(int)
    rr = np.clip(np.stack([i - 1, i - 1, i, i], 1), 0, h - 1)
    cc = np.clip(np.stack([j - 1, j, j - 1, j], 1), 0, w - 1)
    ok = ~tess.occ[rr, cc] & corner[:, None]
    Q = nearest[rr, cc].astype(float)
    V = Q - R[:, None, :]
    Vn = V / np.maximum(np.hypot(V[..., 0], V[..., 1]), 1e-9)[..., None]
    side = tg[:, None, 0] * V[..., 1] - tg[:, None, 1] * V[..., 0]   # < 0 izquierda
    sl = np.where(ok, side, np.inf)
    sr = np.where(ok, side, -np.inf)
    il, ir = np.argmin(sl, 1), np.argmax(sr, 1)
    rows = np.arange(n)
    two = (sl[rows, il] < 0) & (sr[rows, ir] > 0)
    u = Vn[rows, ir] - Vn[rows, il]                                  # hacia la derecha
    un = np.hypot(u[:, 0], u[:, 1])
    u = np.where((two & (un > 1e-6))[:, None], u / np.maximum(un, 1e-9)[:, None], nr)
    # suavizado de la dirección (el contorno está cuantizado al píxel)
    if n >= 3:
        k = np.ones(min(win, n if n % 2 else n - 1))
        us = np.stack([np.convolve(u[:, 0], k, "same"), np.convolve(u[:, 1], k, "same")], 1)
        usn = np.hypot(us[:, 0], us[:, 1])
        u = np.where((usn > 1e-6)[:, None], us / np.maximum(usn, 1e-9)[:, None], u)
    # casi paralela a la ruta → normal
    par = np.abs((u * tg).sum(1)) / tn > math.cos(math.radians(90.0 - min_cross))
    u = np.where(par[:, None], nr, u)
    # orientar hacia la derecha de la ruta
    sgn = np.sign((tg[:, 0] * u[:, 1] - tg[:, 1] * u[:, 0]))
    u = u * np.where(sgn == 0, 1.0, sgn)[:, None]
    ci = np.clip(np.rint(R[:, 0]).astype(int), 0, w - 1)
    ri = np.clip(np.rint(R[:, 1]).astype(int), 0, h - 1)
    half = dt[ri, ci] - np.hypot(R[:, 0] - ci, R[:, 1] - ri) - 0.75 - shrink_px
    half = np.clip(half, 0.0, None)
    half[[0, -1]] = 0.0                                              # S y T degenerados
    QL = R - u * half[:, None]
    QR = R + u * half[:, None]
    return list(zip(QL, QR)), sel


def _subsample(n: int, stride: int) -> np.ndarray:
    """Índices 0, stride, 2·stride, …, n − 1 (siempre incluye el último)."""
    sel = np.arange(0, n, max(1, stride))
    if sel[-1] != n - 1:
        sel = np.append(sel, n - 1)
    return sel


def route_portals(route_pts: np.ndarray, tess: Tessellation, nearest: np.ndarray, shrink_px: float,
                  stride: int = 3, min_angle: float = 0.0, drop_invalid: bool = True,
                  return_index: bool = False, min_cross: float = 45.0, dt: np.ndarray | None = None):
    """Portales a lo largo de una ruta del esqueleto.

    En cada punto p del esqueleto (esquina entre píxeles de teselas distintas)
    se toma el obstáculo más cercano de cada tesela vecina; el segmento entre
    ellos pasa por p y es libre (está dentro del disco libre de radio = holgura).
    Los extremos se acercan ``shrink_px`` hacia p para dejar un margen.
    Con ``dt`` (transformada de distancia exacta), los puntos sin portal de
    teselado válido (conectores, uniones) reciben una cuerda perpendicular a la
    ruta dentro de su disco libre.  S y T dan portales degenerados.
    """
    h, w = tess.labels.shape
    sel = _subsample(len(route_pts), stride)          # los puntos del esqueleto están a ~1 px
    R = np.asarray(route_pts, float)[sel]
    n = len(R)
    stride_eff = max(1, stride)
    fx, fy = R[:, 0] + 0.5, R[:, 1] + 0.5
    corner = (np.abs(fx - np.round(fx)) < 1e-6) & (np.abs(fy - np.round(fy)) < 1e-6)
    corner[[0, -1]] = False
    j = np.round(fx).astype(int)
    i = np.round(fy).astype(int)
    rr = np.stack([i - 1, i - 1, i, i], axis=1)
    cc = np.stack([j - 1, j, j - 1, j], axis=1)
    inb = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rrc, ccc = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
    ok = inb & ~tess.occ[rrc, ccc] & corner[:, None]
    Q = nearest[rrc, ccc].astype(float)                            # (n, 4, 2)
    d = np.zeros_like(R)
    d[1:-1] = R[2:] - R[:-2]
    V = Q - R[:, None, :]
    cr = d[:, None, 0] * V[..., 1] - d[:, None, 1] * V[..., 0]      # (n, 4); < 0 = izquierda
    cr_l = np.where(ok, cr, np.inf)
    cr_r = np.where(ok, cr, -np.inf)
    il = np.argmin(cr_l, axis=1)
    ir = np.argmax(cr_r, axis=1)
    rows = np.arange(n)
    valid = (cr_l[rows, il] < 0) & (cr_r[rows, ir] > 0)
    QL = Q[rows, il]
    QR = Q[rows, ir]
    # Portal TRANSVERSAL: a–p–b casi alineados (ángulo ≥ min_angle).  En las
    # uniones del Voronoi (3+ teselas) la cuerda a–b puede quedar casi
    # paralela a la ruta y obligaría al embudo a desviarse: se descarta.
    if min_angle > 0:
        va, vb = QL - R, QR - R
        cosang = (va * vb).sum(1) / np.maximum(np.hypot(*va.T) * np.hypot(*vb.T), 1e-9)
        valid &= cosang <= math.cos(math.radians(min_angle))
    # tangente suavizada de la ruta (≈ 8 px de base)
    m = max(1, int(round(4 / stride_eff)))
    if n > 2 * m:
        tg = np.zeros_like(R)
        tg[m:-m] = R[2 * m:] - R[:-2 * m]
        tg[:m] = tg[m]
        tg[-m:] = tg[-m - 1]
    else:
        tg = np.zeros_like(R)
        tg[:] = R[-1] - R[0]
    if min_cross > 0:
        # ángulo entre la cuerda y la tangente de la ruta
        ch = QR - QL
        cosx = np.abs((ch * tg).sum(1)) / np.maximum(np.hypot(*ch.T) * np.hypot(*tg.T), 1e-9)
        valid &= cosx <= math.cos(math.radians(min_cross))
    # Los puntos de obstáculo más cercanos están cuantizados al píxel (contorno
    # en escalera): una media móvil a lo largo de la ruta recupera la forma real
    # y evita que el embudo genere un vértice por escalón.  El error (<< 1 px)
    # queda cubierto por el margen de los portales.
    win = max(3, (7 // stride_eff) | 1)               # ≈ 7 px a lo largo de la ruta
    win = min(win, n if n % 2 else n - 1)             # 'same' exige ventana ≤ n
    k = np.ones(max(win, 1))
    wv = np.convolve(valid.astype(float), k, "same")
    for Qx in (QL, QR):
        for c in range(2):
            s = np.convolve(np.where(valid, Qx[:, c], 0.0), k, "same")
            Qx[:, c] = np.where(valid & (wv > 0), s / np.maximum(wv, 1e-9), Qx[:, c])

    def shrink(Qx):
        v = R - Qx
        dn = np.hypot(v[:, 0], v[:, 1])
        f = np.clip(shrink_px / np.maximum(dn, 1e-9), 0, 1)[:, None]
        return Qx + v * f
    QL = np.where(valid[:, None], shrink(QL), R)
    QR = np.where(valid[:, None], shrink(QR), R)
    if dt is not None:
        # Portales de respaldo (conectores, uniones, portales descartados):
        # cuerda perpendicular a la ruta dentro del disco libre de radio
        # ρ = dt(c) − |p − c| − √2/2 (ver FastChecker), menos el margen.
        fb = ~valid
        fb[[0, -1]] = False
        ci = np.clip(np.rint(R[:, 0]).astype(int), 0, w - 1)
        ri = np.clip(np.rint(R[:, 1]).astype(int), 0, h - 1)
        half = dt[ri, ci] - np.hypot(R[:, 0] - ci, R[:, 1] - ri) - 0.75 - shrink_px
        tn = np.hypot(tg[:, 0], tg[:, 1])
        fb &= (half > 0.5) & (tn > 1e-9)
        if fb.any():
            nr = np.stack([-tg[:, 1], tg[:, 0]], 1) / np.maximum(tn, 1e-9)[:, None]   # lado derecho
            QL[fb] = R[fb] - nr[fb] * half[fb, None]
            QR[fb] = R[fb] + nr[fb] * half[fb, None]
            valid = valid | fb
    # Se conservan siempre S y T; los portales no válidos se omiten (la
    # validación final repara el tramo si hiciera falta).
    keep = valid.copy() if drop_invalid else np.ones(n, bool)
    keep[[0, -1]] = True
    if return_index:
        return list(zip(QL[keep], QR[keep])), sel[keep]
    return list(zip(QL[keep], QR[keep]))


# --------------------------------------------------------------------------- #
# 7. Filetes circulares (curvatura acotada)
# --------------------------------------------------------------------------- #
def fillet_path(occ: np.ndarray, P: np.ndarray, R_px: float, step: float = 1.0):
    """Redondea cada esquina con un arco de radio R (o el mayor que quepa sin
    colisión).  Devuelve (ruta densa, radio mínimo usado [px])."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P.copy(), math.inf
    seg = np.hypot(*np.diff(P, axis=0).T)
    out = [P[0]]
    rmin = math.inf
    for i in range(1, len(P) - 1):
        a, v, b = P[i - 1], P[i], P[i + 1]
        u1 = (v - a) / max(seg[i - 1], 1e-12)
        u2 = (b - v) / max(seg[i], 1e-12)
        theta = math.acos(float(np.clip(u1 @ u2, -1, 1)))
        if theta < 1e-3:
            out.append(v)
            continue
        tmax = 0.5 * min(seg[i - 1], seg[i])
        t = min(R_px * math.tan(theta / 2), tmax)
        arc = None
        while t > 0.5:
            r = t / math.tan(theta / 2)
            p1 = v - u1 * t
            p2 = v + u2 * t
            bis = u2 - u1
            bis /= np.hypot(*bis)
            c = v + bis * (r / math.cos(theta / 2))
            a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
            a2 = math.atan2(p2[1] - c[1], p2[0] - c[0])
            da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
            m = max(3, int(abs(da) * r / step) + 1)
            ang = a1 + np.linspace(0, da, m)
            arc = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)
            if _free(occ, np.vstack([out[-1], arc])):
                rmin = min(rmin, r)
                break
            arc = None
            t *= 0.6
        if arc is None:
            out.append(v)                          # esquina sin redondear
            rmin = 0.0
        else:
            out.extend(arc)
    out.append(P[-1])
    return np.asarray(out), rmin


# --------------------------------------------------------------------------- #
# 3. Dijkstra con pesos + alternativas por penalización
# --------------------------------------------------------------------------- #
def _dijkstra(g: SkeletonGraph, s: int, t: int, weight: dict[int, float]):
    dist = {s: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == t:
            break
        if d > dist.get(u, math.inf):
            continue
        for eid in g.adj[u]:
            v = g.other(eid, u)
            nd = d + weight[eid]
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = (u, eid)
                heapq.heappush(heap, (nd, v))
    if t not in dist:
        return None
    nodes, edges = [t], []
    while nodes[-1] != s:
        u, eid = prev[nodes[-1]]
        edges.append(eid)
        nodes.append(u)
    return nodes[::-1], edges[::-1]


def edge_weights(g: SkeletonGraph, mode: str = "cuerda", alpha: float = 0.25,
                 base: dict[int, float] | None = None) -> dict[int, float]:
    """Pesos de las aristas para Dijkstra.  ``longitud``: longitud de la
    polilínea del esqueleto (serpentea: sobreestima la ruta tensa);
    ``cuerda``: cuerda + alpha·(longitud − cuerda), mejor estimador de la
    longitud tras el embudo.  ``base``: pesos ya calculados (caché del mapa);
    sólo se calculan los de las aristas nuevas."""
    w = {}
    for eid, e in g.edges.items():
        if base is not None and eid in base:
            w[eid] = base[eid]
        elif mode == "longitud":
            w[eid] = e.length
        else:
            ch = float(math.hypot(e.pts[-1][0] - e.pts[0][0], e.pts[-1][1] - e.pts[0][1]))
            w[eid] = ch + alpha * (e.length - ch)
    return w


def candidate_routes(g: SkeletonGraph, s: int, t: int, k: int, penalty: float = 1.7,
                     weight: dict[int, float] | None = None):
    weight = dict(weight) if weight is not None else {eid: e.length for eid, e in g.edges.items()}
    seen = set()
    out = []
    for _ in range(max(1, k) * 3):
        r = _dijkstra(g, s, t, weight)
        if r is None:
            break
        # rutas que sólo difieren en el conector de S/T no son alternativas
        skel = [eid for eid in r[1] if g.edges[eid].kind != "connector"]
        key = frozenset(skel)
        if key not in seen:
            seen.add(key)
            out.append(r)
            if len(out) >= k:
                break
        for eid in (skel or r[1]):
            weight[eid] *= penalty
    return out


def route_polyline(g: SkeletonGraph, nodes, edges) -> np.ndarray:
    parts = []
    for i, eid in enumerate(edges):
        pts = g.oriented_pts(eid, nodes[i])
        if g.edges[eid].kind != "skeleton":         # conectores: 1 punto/px (portales de respaldo)
            pts = densify(pts, 1.0)
        parts.append(pts if i == 0 else pts[1:])
    return np.vstack(parts) if parts else np.asarray([g.nodes[nodes[0]]])


# --------------------------------------------------------------------------- #
# Planificador con caché
# --------------------------------------------------------------------------- #
@dataclass
class TVEConfig:
    robot_diameter: float = 0.5
    safety_margin: float = 0.0
    dilation_shape: str = "disk"
    generators: str = "auto"            # auto | obstacles | segments
    segment_size: float = 2.0
    node_merge_dist: float = 3.0
    candidates: int = 4                 # K rutas alternativas
    clearance_extra: float = 0.10       # [m] margen de los portales (holgura extra)
    smoothing: str = "arcos"            # arcos | bezier | ninguno
    turn_radius: float = 1.0            # [m] radio de los filetes
    edge_weight: str = "cuerda"         # cuerda | longitud (pesos de Dijkstra)
    weight_alpha: float = 0.25
    portal_min_angle: float = 0.0       # [°] ángulo mínimo a–p–b (0 = sin filtro)
    portal_min_cross: float = 45.0      # [°] ángulo mínimo entre el portal y la ruta
    portal_mode: str = "ambos"          # teselas (cuerda a–b) | disco (diámetros del disco libre) | ambos
    portal_drop_invalid: bool = True
    multi_attach: bool = True           # varios conectores por sector para S y T
    fast_check: bool = True             # aceptación rápida por transformada de distancia


@dataclass
class TVEResult:
    success: bool
    message: str
    path: np.ndarray | None = None       # ruta final (suavizada)
    taut: np.ndarray | None = None       # ruta del embudo
    start: np.ndarray | None = None
    goal: np.ndarray | None = None
    routes: list = field(default_factory=list)
    portals: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    graph: SkeletonGraph | None = None


class _Prepared:
    __slots__ = ("occ", "tess", "nearest", "graph", "free_comp", "timings", "corners", "checker", "mode",
                 "weights", "spoke_max")


_CACHE: "OrderedDict[tuple, _Prepared]" = OrderedDict()
_CACHE_MAX = 12


def clear_cache():
    _CACHE.clear()


class TVEPlanner:
    def __init__(self, env: Environment, cfg: TVEConfig | None = None):
        self.env = env
        self.cfg = cfg or TVEConfig()
        self.prep: _Prepared | None = None
        self.cache_hit = False

    def prepare(self, use_cache: bool = True) -> _Prepared:
        c = self.cfg
        mpp = self.env.meters_per_pixel
        t0 = time.perf_counter()
        r = robot_radius_px(c.robot_diameter, c.safety_margin, mpp)
        occ = configuration_space(self.env.obstacles, r, c.dilation_shape)
        t1 = time.perf_counter()
        key = (hash(occ.tobytes()), occ.shape, c.generators, round(c.segment_size / mpp, 3), c.node_merge_dist)
        if use_cache and key in _CACHE:
            _CACHE.move_to_end(key)
            self.prep = _CACHE[key]
            self.cache_hit = True
            self.prep_timings = {"dilatacion": t1 - t0}
            return self.prep
        self.cache_hit = False
        seg_px = c.segment_size / mpp
        free_comp = free_components(occ)
        checker = FastChecker(occ)
        modes = ["obstacles", "contours"] if c.generators == "auto" else [c.generators]
        t_tess = time.perf_counter() - t1        # componentes + transformada exacta
        t_skel = 0.0
        best = None
        for mode in modes:
            ta = time.perf_counter()
            tess, nearest = voronoi_tessellation(occ, mode, seg_px)
            tb = time.perf_counter()
            graph = extract_skeleton(tess, c.node_merge_dist)
            tc = time.perf_counter()
            t_tess += tb - ta
            t_skel += tc - tb
            bad = skeleton_defects(graph, occ, free_comp)
            if best is None or bad < best[0]:
                best = (bad, mode, tess, nearest, graph)
            if bad == 0:
                break
        # modo automático: si con un generador por obstáculo el esqueleto no
        # cubre (o parte) alguna región libre —p.ej. un laberinto de un único
        # obstáculo— se divide el contorno en tramos; así no hacen falta
        # puentes costosos en cada consulta.
        bad, mode, tess, nearest, graph = best
        p = _Prepared()
        p.occ, p.tess, p.nearest, p.graph, p.checker = occ, tess, nearest, graph, checker
        p.free_comp = free_comp
        p.corners = build_corner_index(graph)
        p.spoke_max = 2.0 * float(tess.clearance.max()) + 4.0
        p.mode = mode
        p.weights = {"key": (c.edge_weight, c.weight_alpha), "w": edge_weights(graph, c.edge_weight, c.weight_alpha)}
        p.timings = {"teselado": t_tess, "esqueleto": (time.perf_counter() - t1) - t_tess}
        self.prep = p
        self.prep_timings = {"dilatacion": t1 - t0, **p.timings}
        if use_cache:
            _CACHE[key] = p
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        return p

    def plan(self, start_xy, goal_xy, use_cache: bool = True) -> TVEResult:
        from common.problem import snap_free
        c = self.cfg
        mpp = self.env.meters_per_pixel
        p = self.prepare(use_cache)
        timings = dict(self.prep_timings)
        S, _ = snap_free(p.occ, start_xy)
        T, _ = snap_free(p.occ, goal_xy)
        occ = p.checker if c.fast_check else p.occ      # aceptación rápida + exacta, o sólo exacta
        res = TVEResult(False, "", start=S, goal=T, timings=timings)
        if p.free_comp[int(S[1]), int(S[0])] != p.free_comp[int(T[1]), int(T[0])]:
            res.message = "No existe ruta: S y T están en regiones libres desconectadas"
            return res
        # --- conexión de S y T (copia del grafo en caché)
        t0 = time.perf_counter()
        g = p.graph.light_copy()
        if c.multi_attach:
            refs_s, refs_t = find_attach_multi(p, S), find_attach_multi(p, T)
        else:
            refs_s, refs_t = ([r] if (r := find_attach(p, S)) else []), ([r] if (r := find_attach(p, T)) else [])
        if refs_s and refs_t:
            s_node, t_node = attach_multi(g, [(S, "start", refs_s), (T, "goal", refs_t)])
        else:                                               # respaldo: BFS por el espacio libre
            s_node, _ = attach_point(g, p.tess, S, "start")
            t_node, _ = attach_point(g, p.tess, T, "goal")
        comp = g.components()
        if comp[s_node] != comp[t_node]:
            bridge_components(g, p.tess, s_node, t_node)
        timings["conexion_ST"] = time.perf_counter() - t0
        res.graph = g
        # --- rutas candidatas
        t0 = time.perf_counter()
        wkey = (c.edge_weight, c.weight_alpha)
        if p.weights.get("key") != wkey:
            p.weights = {"key": wkey, "w": edge_weights(p.graph, c.edge_weight, c.weight_alpha)}
        weight = edge_weights(g, c.edge_weight, c.weight_alpha, base=p.weights["w"])
        cands = candidate_routes(g, s_node, t_node, c.candidates, weight=weight)
        timings["rutas"] = time.perf_counter() - t0
        if not cands:
            res.message = "El esqueleto no conecta S y T"
            return res
        # --- embudo sobre cada candidata
        t0 = time.perf_counter()
        shrink = 1.0 + c.clearance_extra / mpp
        # Embudo sobre cada candidata (barato); la validación exacta y la
        # simplificación se hacen sólo sobre la ganadora.
        evals = []
        skipped = 0
        for nodes, edges in cands:
            rp = route_polyline(g, nodes, edges)
            rp[0], rp[-1] = S, T
            variants = []
            if c.portal_mode in ("teselas", "ambos"):
                variants.append(route_portals(rp, p.tess, p.nearest, shrink, min_angle=c.portal_min_angle,
                                              drop_invalid=c.portal_drop_invalid, return_index=True,
                                              min_cross=c.portal_min_cross, dt=p.checker.dt))
            if c.portal_mode in ("disco", "ambos"):
                variants.append(disc_portals(rp, p.tess, p.nearest, p.checker.dt, shrink,
                                             min_cross=c.portal_min_cross))
            Lbest = math.nan
            for portals, ridx in variants:
                # poda: si ni siquiera la cota inferior mejora la mejor, no hay embudo
                if evals and portal_lower_bound(portals, S, T) >= min(e[0] for e in evals):
                    skipped += 1
                    continue
                taut, idx = funnel(portals)
                L = polyline_length(taut)
                evals.append((L, taut, [int(ridx[i]) for i in idx], rp, portals))
                Lbest = L if math.isnan(Lbest) else min(Lbest, L)
            res.routes.append({"polyline": rp, "length_px": Lbest})
        evals.sort(key=lambda e: e[0])
        repairs = fallbacks = 0
        best = None
        # Se validan (y podan) las 4 más cortas: la poda con atajos puede
        # acortar bastante un embudo cuyos portales no eran óptimos, así que la
        # longitud antes de validar no basta para descartar.
        for j, (L, taut, idx, rp, portals) in enumerate(evals[:4]):
            if not _free(occ, taut):
                repairs += 1
                taut = _repair(occ, taut, idx, rp,
                               lambda i0, i1, a, b, rp=rp: _refunnel(occ, p.tess, p.nearest, shrink, rp, i0, i1, a, b))
            taut = prune_vertices(occ, taut)                  # DP + atajos (pocos vértices)
            Lt = polyline_length(taut)
            # Salvaguarda: en zonas muy abiertas el corredor del esqueleto puede
            # dar un embudo casi tan largo como el propio esqueleto; entonces se
            # compara con el esqueleto simplificado por visibilidad.
            if Lt > 0.9 * polyline_length(rp):
                alt = prune_vertices(occ, simplify_collision_free(occ, rp))
                La = polyline_length(alt)
                if La < Lt:
                    taut, Lt = alt, La
                    fallbacks += 1
            if best is None or Lt < best[0]:
                best = (Lt, taut, portals)
        timings["embudo"] = time.perf_counter() - t0
        res.taut, res.portals = best[1], best[2]
        # --- suavizado
        t0 = time.perf_counter()
        if c.smoothing == "arcos":
            path, rmin = fillet_path(occ, res.taut, c.turn_radius / mpp)
            res.stats["radio_filete_min_m"] = rmin * mpp if math.isfinite(rmin) else float("inf")
        elif c.smoothing == "bezier":
            path = smooth_piecewise(p.occ, res.taut).curve
        else:
            path = res.taut
        if not _free(occ, path):
            path = res.taut
        timings["suavizado"] = time.perf_counter() - t0
        res.path = path
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        res.stats.update({"candidatas": len(cands), "podadas_cota": skipped, "reparaciones": repairs,
                          "respaldo_visibilidad": fallbacks,
                          "cache": self.cache_hit,
                          "nodos_esqueleto": len(p.graph.nodes), "vertices_embudo": len(res.taut)})
        res.success = True
        res.message = "Ruta encontrada" + (" (esqueleto en caché)" if self.cache_hit else "")
        return res


def _refunnel(occ, tess, nearest, shrink_px, rp, i0, i1, a, b):
    """Reparación local: embudo sobre el tramo rp[i0..i1] con TODOS los
    portales (sin submuestreo ni filtros) y margen creciente."""
    sub = np.vstack([a, rp[i0:i1 + 1], b])
    if len(sub) < 3:
        return None
    for mult in (1.0, 2.0, 3.5):
        if isinstance(occ, FastChecker):
            prt, _ = disc_portals(sub, tess, nearest, occ.dt, shrink_px * mult, stride=1)
        else:
            prt = route_portals(sub, tess, nearest, shrink_px * mult, stride=1, min_angle=0.0, min_cross=0.0)
        path, _ = funnel(prt)
        if _free(occ, path):
            return path
    return None


def _repair(occ, taut, idx, rp, refunnel=None):
    """Si algún tramo del embudo roza un obstáculo (portales muy curvos), se
    reemplaza por la subruta del esqueleto simplificada con cuerdas libres.

    ``idx``: índice EN LA RUTA DEL ESQUELETO ``rp`` del portal de cada vértice.
    El vértice está sobre la cuerda de su portal, dentro del disco libre de su
    punto del esqueleto: a → rp[i0] … rp[i1] → b es una polilínea libre."""
    out = [taut[0]]
    for k in range(len(taut) - 1):
        a, b = taut[k], taut[k + 1]
        if _free(occ, np.vstack([a, b])):
            out.append(b)
            continue
        i0, i1 = idx[k], idx[k + 1]
        fixed = refunnel(i0, i1, a, b) if refunnel is not None else None
        if fixed is None:
            fixed = simplify_collision_free(occ, np.vstack([a, rp[i0:i1 + 1], b]))
        out.extend(fixed[1:])
    out = np.asarray(out)
    if not _free(occ, out):
        out = simplify_collision_free(occ, rp)
    return out
