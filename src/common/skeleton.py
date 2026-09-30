"""Obtención de fronteras y nodos (bloque "Obtaining borders and nodes").

El esqueleto es el conjunto de fronteras entre teselas.  Se representa de
forma exacta sobre la malla de *esquinas* de píxel: una "grieta" (lado común
entre dos píxeles vecinos) pertenece a la frontera si ambos píxeles tienen
etiquetas distintas.  En cada esquina confluyen 0, 2, 3 o 4 grietas de
frontera; las esquinas con 3 o más son los **nodos** del artículo (puntos
donde se intersectan tres o más teselas).  Las cadenas de grietas entre nodos
son las **aristas** (trayectorias) del grafo.

Una esquina (i, j) está en la posición ``(x, y) = (j - 0.5, i - 0.5)``.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from common.geometry import bilinear, polyline_length, segment_free
from common.tessellation import Tessellation


# --------------------------------------------------------------------------- #
# Estructura de grafo (multigrafo no dirigido con geometría en las aristas)
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    u: int
    v: int
    pts: np.ndarray                  # (N, 2) polilínea orientada de u a v
    tiles: tuple = (0, 0)            # teselas que separa
    kind: str = "skeleton"           # skeleton | connector | bridge
    length: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.pts = np.asarray(self.pts, dtype=float)
        self.length = geometric_length(self.pts)


def geometric_length(pts: np.ndarray, eps: float = 0.75) -> float:
    """Longitud euclidiana aproximada de una polilínea en escalera.

    Las fronteras siguen los lados de los píxeles (distancia Manhattan); se
    simplifican con Douglas-Peucker (tolerancia < 1 px) antes de medirlas.
    """
    if len(pts) < 3:
        return polyline_length(pts)
    approx = cv2.approxPolyDP(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), eps, False)
    return polyline_length(approx.reshape(-1, 2))


class SkeletonGraph:
    def __init__(self) -> None:
        self.nodes: dict[int, np.ndarray] = {}
        self.node_kind: dict[int, str] = {}
        self.edges: dict[int, Edge] = {}
        self.adj: dict[int, set[int]] = {}
        self._next_node = 0
        self._next_edge = 0

    # ----------------------------------------------------------- mutations
    def add_node(self, xy, kind: str = "junction") -> int:
        nid = self._next_node
        self._next_node += 1
        self.nodes[nid] = np.asarray(xy, dtype=float)
        self.node_kind[nid] = kind
        self.adj[nid] = set()
        return nid

    def add_edge(self, u: int, v: int, pts, tiles=(0, 0), kind: str = "skeleton") -> int:
        eid = self._next_edge
        self._next_edge += 1
        self.edges[eid] = Edge(u, v, pts, tuple(tiles), kind)
        self.adj[u].add(eid)
        self.adj[v].add(eid)
        return eid

    def remove_edge(self, eid: int) -> None:
        e = self.edges.pop(eid)
        self.adj[e.u].discard(eid)
        self.adj[e.v].discard(eid)

    def remove_node(self, nid: int) -> None:
        for eid in list(self.adj[nid]):
            self.remove_edge(eid)
        del self.adj[nid]
        del self.nodes[nid]
        del self.node_kind[nid]

    # -------------------------------------------------------------- queries
    def other(self, eid: int, nid: int) -> int:
        e = self.edges[eid]
        return e.v if e.u == nid else e.u

    def oriented_pts(self, eid: int, start: int) -> np.ndarray:
        e = self.edges[eid]
        return e.pts if e.u == start else e.pts[::-1]

    def degree(self, nid: int) -> int:
        return len(self.adj[nid])

    def components(self) -> dict[int, int]:
        comp: dict[int, int] = {}
        c = 0
        for s in self.nodes:
            if s in comp:
                continue
            comp[s] = c
            dq = deque([s])
            while dq:
                n = dq.popleft()
                for eid in self.adj[n]:
                    m = self.other(eid, n)
                    if m not in comp:
                        comp[m] = c
                        dq.append(m)
            c += 1
        return comp

    def light_copy(self) -> "SkeletonGraph":
        """Copia superficial: comparte los objetos Edge y las coordenadas (que las
        consultas no modifican; ``split_edge`` crea aristas nuevas)."""
        g = SkeletonGraph()
        g.nodes = dict(self.nodes)
        g.node_kind = dict(self.node_kind)
        g.edges = dict(self.edges)
        g.adj = {k: set(v) for k, v in self.adj.items()}
        g._next_node = self._next_node
        g._next_edge = self._next_edge
        return g

    def copy(self) -> "SkeletonGraph":
        g = SkeletonGraph()
        g.nodes = {k: v.copy() for k, v in self.nodes.items()}
        g.node_kind = dict(self.node_kind)
        g.edges = {k: Edge(e.u, e.v, e.pts.copy(), e.tiles, e.kind) for k, e in self.edges.items()}
        g.adj = {k: set(v) for k, v in self.adj.items()}
        g._next_node = self._next_node
        g._next_edge = self._next_edge
        return g

    # --------------------------------------------------------- operations
    def split_edge(self, eid: int, k: int, kind: str = "split") -> int:
        """Divide la arista en el índice ``k`` de su polilínea; devuelve el nodo."""
        e = self.edges[eid]
        n = len(e.pts)
        if k <= 0:
            return e.u
        if k >= n - 1:
            return e.v
        nid = self.add_node(e.pts[k], kind)
        self.remove_edge(eid)
        self.add_edge(e.u, nid, e.pts[:k + 1], e.tiles, e.kind)
        self.add_edge(nid, e.v, e.pts[k:], e.tiles, e.kind)
        return nid

    def merge_nodes(self, keep: int, drop: int, via: int) -> None:
        """Contrae la arista ``via`` (keep<->drop) manteniendo la geometría."""
        path = self.oriented_pts(via, keep)  # keep -> drop
        self.remove_edge(via)
        for eid in list(self.adj[drop]):
            e = self.edges[eid]
            pts = e.pts
            if e.u == drop:
                pts = np.vstack([path[:-1], pts])
                e.u = keep
            if e.v == drop:
                pts = np.vstack([pts, path[::-1][1:]])
                e.v = keep
            e.pts = pts
            e.length = geometric_length(pts)
            self.adj[drop].discard(eid)
            self.adj[keep].add(eid)
        del self.adj[drop]
        del self.nodes[drop]
        del self.node_kind[drop]
        # los lazos (u == v) nunca forman parte de una ruta simple
        for eid in list(self.adj[keep]):
            if self.edges[eid].u == self.edges[eid].v:
                self.remove_edge(eid)


# --------------------------------------------------------------------------- #
# Extracción del esqueleto a partir del teselado
# --------------------------------------------------------------------------- #
_DI = (0, 0, 1, -1)   # right, left, down, up
_DJ = (1, -1, 0, 0)
_REV = (1, 0, 3, 2)


def extract_skeleton(tess: Tessellation, merge_dist: float = 3.0) -> SkeletonGraph:
    L = tess.labels
    occ = tess.occ
    C = tess.comp_of_label[np.clip(L, 0, len(tess.comp_of_label) - 1)]
    h, w = L.shape
    W1 = w + 1
    # Grieta de frontera: etiquetas distintas, salvo que ambos píxeles sean
    # del mismo obstáculo o uno sea obstáculo y el otro una tesela del mismo
    # obstáculo (solo ocurre en el modo "segments").
    HB = np.zeros((h + 1, w + 1), dtype=bool)
    HB[1:h, 0:w] = (L[:-1, :] != L[1:, :]) & ~((occ[:-1, :] | occ[1:, :]) & (C[:-1, :] == C[1:, :]))
    VB = np.zeros((h + 1, w + 1), dtype=bool)
    VB[0:h, 1:w] = (L[:, :-1] != L[:, 1:]) & ~((occ[:, :-1] | occ[:, 1:]) & (C[:, :-1] == C[:, 1:]))
    right = HB
    left = np.zeros_like(HB)
    left[:, 1:] = HB[:, :-1]
    down = VB
    up = np.zeros_like(VB)
    up[1:, :] = VB[:-1, :]
    deg = right.astype(np.int8) + left + down + up

    has = [right.ravel().tolist(), left.ravel().tolist(), down.ravel().tolist(), up.ravel().tolist()]
    node_mask = (deg >= 3) | (deg == 1)
    is_node = node_mask.ravel().tolist()
    usedH = np.zeros(HB.size, dtype=bool)
    usedV = np.zeros(VB.size, dtype=bool)

    def crack_id(i, j, d):
        if d == 0:
            return 0, i * W1 + j
        if d == 1:
            return 0, i * W1 + j - 1
        if d == 2:
            return 1, i * W1 + j
        return 1, (i - 1) * W1 + j

    def tiles_of(i, j, d):
        kind, cid = crack_id(i, j, d)
        ci, cj = divmod(cid, W1)
        if kind == 0:
            return int(L[ci - 1, cj]), int(L[ci, cj])
        return int(L[ci, cj - 1]), int(L[ci, cj])

    def mark(i, j, d):
        kind, cid = crack_id(i, j, d)
        (usedH if kind == 0 else usedV)[cid] = True

    def is_used(i, j, d):
        kind, cid = crack_id(i, j, d)
        return bool((usedH if kind == 0 else usedV)[cid])

    def walk(i0, j0, d0):
        pts = [(i0, j0)]
        i, j, d = i0, j0, d0
        while True:
            mark(i, j, d)
            i += _DI[d]
            j += _DJ[d]
            pts.append((i, j))
            f = i * W1 + j
            if is_node[f] or (i == i0 and j == j0):
                return pts
            rev = _REV[d]
            nd = -1
            for cand in range(4):
                if cand != rev and has[cand][f]:
                    nd = cand
                    break
            if nd < 0:        # no debería ocurrir (grado 1 imposible)
                return pts
            d = nd

    def to_xy(pts):
        a = np.asarray(pts, dtype=float)
        return np.column_stack([a[:, 1] - 0.5, a[:, 0] - 0.5])

    g = SkeletonGraph()
    node_of: dict[int, int] = {}
    degf = deg.ravel()
    node_idx = np.flatnonzero(node_mask.ravel())
    for f in node_idx:
        i, j = divmod(int(f), W1)
        node_of[int(f)] = g.add_node((j - 0.5, i - 0.5), "junction" if degf[f] >= 3 else "end")

    for f in node_idx:
        i0, j0 = divmod(int(f), W1)
        for d in range(4):
            if has[d][f] and not is_used(i0, j0, d):
                tiles = tiles_of(i0, j0, d)
                pts = walk(i0, j0, d)
                i1, j1 = pts[-1]
                u = node_of[int(f)]
                v = node_of.get(i1 * W1 + j1)
                if v is None:  # seguridad: cadena abierta
                    v = g.add_node((j1 - 0.5, i1 - 0.5), "end")
                    node_of[i1 * W1 + j1] = v
                if u == v and len(pts) <= 2:
                    continue
                g.add_edge(u, v, to_xy(pts), tiles)

    # Fronteras cerradas sin nodos (p.ej. un único obstáculo dentro del marco)
    for cid in np.flatnonzero(HB.ravel() & ~usedH):
        if usedH[cid]:
            continue
        i0, j0 = divmod(int(cid), W1)
        tiles = tiles_of(i0, j0, 0)
        pts = walk(i0, j0, 0)
        xy = to_xy(pts)
        m = len(xy) - 1
        if m < 4:
            continue
        a = g.add_node(xy[0], "loop")
        b = g.add_node(xy[m // 2], "loop")
        g.add_edge(a, b, xy[: m // 2 + 1], tiles)
        g.add_edge(b, a, xy[m // 2:], tiles)

    # Los lazos u==v producidos por cadenas que regresan al mismo nodo
    for eid in [e for e, ed in g.edges.items() if ed.u == ed.v]:
        e = g.edges[eid]
        m = len(e.pts) - 1
        if m >= 4:
            mid = g.split_edge(eid, m // 2, "loop")
            g.node_kind[mid] = "loop"
        else:
            g.remove_edge(eid)

    if any(k == "end" for k in g.node_kind.values()):
        prune_spurs(g, tess)
        dissolve_degree2(g)
    contract_short_edges(g, merge_dist)
    return g


def prune_spurs(g: SkeletonGraph, tess: Tessellation, factor: float = 1.6, slack: float = 3.0) -> int:
    """Elimina ramas que solo unen el eje medial con la pared.

    Una rama hoja que termina en un obstáculo y cuya longitud no supera
    ~ la holgura de su nodo de origen es la bisectriz entre dos tramos de la
    misma pared: no aporta caminos nuevos.  Los callejones largos (pasillos
    sin salida) se conservan.
    """
    removed = 0
    changed = True
    while changed:
        changed = False
        for nid in list(g.nodes):
            if nid not in g.nodes or g.node_kind[nid] != "end" or g.degree(nid) != 1:
                continue
            eid = next(iter(g.adj[nid]))
            other = g.other(eid, nid)
            clr = float(bilinear(tess.clearance, g.nodes[other][None, :])[0])
            if g.edges[eid].length <= factor * clr + slack:
                g.remove_node(nid)
                removed += 1
                changed = True
                if g.degree(other) == 0 and g.node_kind[other] in ("junction", "end"):
                    g.remove_node(other)
    for nid in [n for n in g.nodes if g.degree(n) == 0 and g.node_kind[n] in ("junction", "end")]:
        g.remove_node(nid)
    return removed


def dissolve_degree2(g: SkeletonGraph) -> None:
    """Une las dos aristas de los nodos 'junction' que quedaron con grado 2."""
    for nid in list(g.nodes):
        if nid not in g.nodes or g.node_kind[nid] != "junction" or g.degree(nid) != 2:
            continue
        e1, e2 = list(g.adj[nid])
        a = g.other(e1, nid)
        b = g.other(e2, nid)
        if a == nid or b == nid or a == b:
            continue
        p1 = g.oriented_pts(e1, a)       # a -> nid
        p2 = g.oriented_pts(e2, nid)     # nid -> b
        tiles = g.edges[e1].tiles
        g.remove_node(nid)
        g.add_edge(a, b, np.vstack([p1, p2[1:]]), tiles)


def contract_short_edges(g: SkeletonGraph, min_len: float) -> None:
    """Agrupa nodos muy cercanos (equivalente al agrupamiento de Reduccion.m)."""
    if min_len <= 0:
        return
    changed = True
    while changed:
        changed = False
        for eid in sorted(g.edges, key=lambda k: g.edges[k].length):
            e = g.edges.get(eid)
            if e is None or e.kind != "skeleton" or e.length >= min_len:
                continue
            if e.u == e.v:
                g.remove_edge(eid)
                changed = True
                continue
            # nunca contraer si ambos son nodos especiales (inicio/meta)
            if g.node_kind[e.u] in ("start", "goal") and g.node_kind[e.v] in ("start", "goal"):
                continue
            keep, drop = (e.u, e.v) if g.degree(e.u) >= g.degree(e.v) else (e.v, e.u)
            if g.node_kind[drop] in ("start", "goal"):
                keep, drop = drop, keep
            g.merge_nodes(keep, drop, eid)
            changed = True
            break


# --------------------------------------------------------------------------- #
# Conexión de puntos (S, T) y reparación de conectividad
# --------------------------------------------------------------------------- #
def _corner_index(g: SkeletonGraph) -> tuple[np.ndarray, list]:
    """Tabla de todos los puntos del esqueleto: coords y referencia (eid, k)."""
    coords = []
    refs = []
    for eid, e in g.edges.items():
        if e.kind != "skeleton":
            continue
        n = len(e.pts)
        coords.append(e.pts)
        refs.extend((eid, k) for k in range(n))
    if not coords:
        return np.zeros((0, 2)), []
    return np.vstack(coords), refs


def _node_at_ref(g: SkeletonGraph, ref, kind="split") -> int:
    if ref[0] == "node":
        return ref[1]
    eid, k = ref
    return g.split_edge(eid, k, kind)


def attach_point(g: SkeletonGraph, tess: Tessellation, p_xy, kind: str,
                 step: float = 0.25, max_straight_candidates: int = 40) -> tuple[int, bool]:
    """Agrega el punto ``p_xy`` como nodo y lo conecta al esqueleto.

    Primero intenta una recta libre de colisión al punto del esqueleto más
    cercano; si no existe, usa una búsqueda en anchura (BFS) sobre el espacio
    libre hasta tocar el esqueleto.  Devuelve (id_nodo, conectado).
    """
    p = np.asarray(p_xy, dtype=float)
    nid = g.add_node(p, kind)
    coords, refs = _corner_index(g)
    if len(coords) == 0:
        return nid, False
    occ = tess.occ

    d = np.hypot(coords[:, 0] - p[0], coords[:, 1] - p[1])
    order = np.argsort(d)[:max_straight_candidates]
    for idx in order:
        if segment_free(occ, p, coords[idx], step):
            target = _node_at_ref(g, refs[idx])
            g.add_edge(nid, target, np.vstack([p, g.nodes[target]]), kind="connector")
            return nid, True

    # BFS en el espacio libre hasta tocar una esquina del esqueleto
    h, w = occ.shape
    key_to_idx = {}
    for idx, (x, y) in enumerate(coords):
        key_to_idx[(int(round(y + 0.5)), int(round(x + 0.5)))] = idx
    r0, c0 = int(round(p[1])), int(round(p[0]))
    if not (0 <= r0 < h and 0 <= c0 < w) or occ[r0, c0]:
        return nid, False
    parent = {(r0, c0): None}
    dq = deque([(r0, c0)])
    found = None
    while dq and found is None:
        r, c = dq.popleft()
        best = None
        for ci, cj in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)):
            idx = key_to_idx.get((ci, cj))
            if idx is not None:
                dd = np.hypot(coords[idx, 0] - p[0], coords[idx, 1] - p[1])
                if best is None or dd < best[0]:
                    best = (dd, idx)
        if best is not None:
            found = ((r, c), best[1])
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not occ[rr, cc] and (rr, cc) not in parent:
                    parent[(rr, cc)] = (r, c)
                    dq.append((rr, cc))
    if found is None:
        return nid, False
    (r, c), idx = found
    path = []
    cur = (r, c)
    while cur is not None:
        path.append((cur[1], cur[0]))
        cur = parent[cur]
    path = path[::-1]
    target = _node_at_ref(g, refs[idx])
    pts = np.vstack([p, np.asarray(path[1:], dtype=float).reshape(-1, 2), g.nodes[target]])
    g.add_edge(nid, target, pts, kind="connector")
    return nid, True


def free_components(occ: np.ndarray) -> np.ndarray:
    _, comp = cv2.connectedComponents((~occ).astype(np.uint8), connectivity=8)
    return comp


def pixel_of(occ: np.ndarray, xy) -> tuple[int, int] | None:
    """Píxel libre asociado a un punto (esquina o centro)."""
    h, w = occ.shape
    x, y = float(xy[0]), float(xy[1])
    cands = {(int(np.floor(y + 0.5 + dy)), int(np.floor(x + 0.5 + dx)))
             for dx in (-1e-6, 1e-6) for dy in (-1e-6, 1e-6)}
    for r, c in sorted(cands):
        if 0 <= r < h and 0 <= c < w and not occ[r, c]:
            return r, c
    return None


def bridge_components(g: SkeletonGraph, tess: Tessellation, s: int, t: int,
                      max_bridges: int = 50) -> int:
    """Une componentes desconectadas del esqueleto dentro del mismo espacio libre.

    Un teselado por obstáculos no genera frontera en regiones delimitadas por
    un único obstáculo (p.ej. un pasillo formado solo por el marco).  En ese
    caso el grafo puede quedar partido aunque exista camino.  Se agregan
    "puentes" con Dijkstra sobre la malla, penalizando la cercanía a
    obstáculos para que sigan el centro del pasillo.  Devuelve nº de puentes.
    """
    occ = tess.occ
    h, w = occ.shape
    clearance = tess.clearance
    n_bridges = 0
    while n_bridges < max_bridges:
        comp = g.components()
        if comp[s] == comp[t]:
            return n_bridges
        src_comp = comp[s]
        # Anclas: píxeles libres pegados a cada punto del grafo
        anchor_comp = np.full((h, w), -1, dtype=np.int32)
        anchor_ref: dict[tuple[int, int], tuple] = {}
        for eid, e in g.edges.items():
            cc = comp[e.u]
            for k, pt in enumerate(e.pts):
                if e.kind != "skeleton" and 0 < k < len(e.pts) - 1:
                    continue
                pix = pixel_of(occ, pt)
                if pix is None:
                    continue
                if anchor_comp[pix] == -1 or cc == src_comp:
                    anchor_comp[pix] = cc
                    if k == 0:
                        anchor_ref[pix] = ("node", e.u)
                    elif k == len(e.pts) - 1:
                        anchor_ref[pix] = ("node", e.v)
                    else:
                        anchor_ref[pix] = (eid, k)
        for nid, xy in g.nodes.items():
            if g.degree(nid) == 0:
                pix = pixel_of(occ, xy)
                if pix is not None:
                    anchor_comp[pix] = comp[nid]
                    anchor_ref[pix] = ("node", nid)

        # Dijkstra multi-fuente desde la componente de S
        dist = np.full((h, w), np.inf)
        parent = {}
        heap = []
        for (r, c), cc in ((k, anchor_comp[k]) for k in anchor_ref):
            if cc == src_comp:
                dist[r, c] = 0.0
                parent[(r, c)] = None
                heap.append((0.0, r, c))
        heapq.heapify(heap)
        goal = None
        while heap:
            dcur, r, c = heapq.heappop(heap)
            if dcur > dist[r, c]:
                continue
            ac = anchor_comp[r, c]
            if ac != -1 and ac != src_comp:
                goal = (r, c)
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc2 = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc2 < w) or occ[rr, cc2]:
                        continue
                    step = 1.4142135623730951 if dr and dc else 1.0
                    nd = dcur + step * (1.0 + 4.0 / (1.0 + clearance[rr, cc2]))
                    if nd < dist[rr, cc2]:
                        dist[rr, cc2] = nd
                        parent[(rr, cc2)] = (r, c)
                        heapq.heappush(heap, (nd, rr, cc2))
        if goal is None:
            return n_bridges
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path = path[::-1]
        a_ref = anchor_ref[path[0]]
        b_ref = anchor_ref[path[-1]]
        a = _node_at_ref(g, a_ref, "bridge")
        # si b estaba en la misma arista (imposible: componentes distintas)
        b = _node_at_ref(g, b_ref, "bridge")
        mid = np.asarray([(c, r) for r, c in path], dtype=float)
        pts = np.vstack([g.nodes[a], mid, g.nodes[b]])
        g.add_edge(a, b, pts, kind="bridge")
        n_bridges += 1
    return n_bridges


def edge_weights(g: SkeletonGraph, tess: Tessellation) -> dict[int, np.ndarray]:
    """Peso (holgura en px) en cada punto de cada arista del esqueleto."""
    return {eid: bilinear(tess.clearance, e.pts) for eid, e in g.edges.items()}
