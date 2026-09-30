"""Utilidades geométricas: verificación de colisión, polilíneas y rayos.

Los píxeles ocupados se consideran cuadrados *abiertos*: un punto que cae
exactamente sobre el borde entre un píxel libre y uno ocupado se considera
libre (contacto sin penetración).  Esto hace consistente la verificación para
puntos del esqueleto, que viven en las esquinas de los píxeles.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

_EPS = 1e-6


def points_in_collision(occ: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Máscara booleana: qué puntos ``(x, y)`` caen dentro de un obstáculo."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    h, w = occ.shape
    x = pts[:, 0]
    y = pts[:, 1]
    out = (x < -0.5) | (y < -0.5) | (x > w - 0.5) | (y > h - 0.5)
    # Una muestra colisiona solo si todos los píxeles que la contienen
    # (1, 2 o 4 cuando cae sobre un borde) están ocupados.
    x0 = np.floor(x + (0.5 - _EPS)).astype(np.int64)
    x1 = np.floor(x + (0.5 + _EPS)).astype(np.int64)
    y0 = np.floor(y + (0.5 - _EPS)).astype(np.int64)
    y1 = np.floor(y + (0.5 + _EPS)).astype(np.int64)
    np.clip(x0, 0, w - 1, out=x0)
    np.clip(x1, 0, w - 1, out=x1)
    np.clip(y0, 0, h - 1, out=y0)
    np.clip(y1, 0, h - 1, out=y1)
    res = occ[y0, x0]
    edge_x = x0 != x1
    edge_y = y0 != y1
    if edge_x.any() or edge_y.any():
        res = res & occ[y0, x1] & occ[y1, x0] & occ[y1, x1]
    return res | out


def _grid_crossing(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Parámetro t∈(0,1) donde a + t·d cruza una línea de malla (k + 0.5).

    Requiere |d| < 1 (a lo sumo un cruce).  NaN si no hay cruce.
    """
    b = a + d
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    line = np.floor(hi - 0.5) + 0.5
    valid = (line > lo) & (line < hi) & (d != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (line - a) / d
    return np.where(valid, t, np.nan)


def polyline_free(occ: np.ndarray, pts: np.ndarray, step: float | None = None) -> bool:
    """Prueba EXACTA de colisión de una polilínea contra píxeles ocupados.

    Cada tramo se parte en los puntos donde cruza las líneas de la malla;
    cada subtramo queda dentro de un único píxel (o sobre un borde) y se
    verifica su punto medio.  Además se detecta el paso por una esquina entre
    dos píxeles ocupados en diagonal.  Los píxeles son cuadrados abiertos:
    rozar un borde no es colisión, atravesar el interior sí.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return True
    if points_in_collision(occ, pts).any():
        return False
    if len(pts) == 1:
        return True
    pts = densify(pts, 0.5)
    A = pts[:-1]
    d = pts[1:] - A
    tx = _grid_crossing(A[:, 0], d[:, 0])
    ty = _grid_crossing(A[:, 1], d[:, 1])
    n = len(A)
    T = np.stack([np.zeros(n), np.nan_to_num(tx, nan=1.0), np.nan_to_num(ty, nan=1.0), np.ones(n)], axis=1)
    T.sort(axis=1)
    mids = 0.5 * (T[:, :-1] + T[:, 1:])                       # (n, 3)
    P = A[:, None, :] + mids[:, :, None] * d[:, None, :]
    if points_in_collision(occ, P.reshape(-1, 2)).any():
        return False
    # paso exacto por una esquina: no se permite pasar entre dos píxeles
    # ocupados que se tocan en diagonal
    corner = ~np.isnan(tx) & ~np.isnan(ty) & (np.abs(tx - ty) < 1e-9)
    if corner.any():
        h, w = occ.shape
        C = A[corner] + tx[corner, None] * d[corner]
        sx = np.sign(d[corner, 0])
        sy = np.sign(d[corner, 1])
        bx = np.clip(np.rint(C[:, 0] + 0.5 * sx).astype(int), 0, w - 1)
        by = np.clip(np.rint(C[:, 1] - 0.5 * sy).astype(int), 0, h - 1)
        cx = np.clip(np.rint(C[:, 0] - 0.5 * sx).astype(int), 0, w - 1)
        cy = np.clip(np.rint(C[:, 1] + 0.5 * sy).astype(int), 0, h - 1)
        if (occ[by, bx] & occ[cy, cx]).any():
            return False
    return True


class FastChecker:
    """Chequeo de colisión con *aceptación rápida* por transformada de distancia.

    ``dt`` = distancia EXACTA (máscara precisa) de cada centro de píxel libre al
    centro de píxel ocupado más cercano.  Para una muestra p redondeada al
    píxel c, la distancia de p a cualquier cuadrado ocupado es
    ρ(p) ≥ dt(c) − |p − c| − √2/2  (desigualdad triangular).  Las bolas abiertas
    B(p, ρ) son libres; si cada par de muestras consecutivas cumple
    ρ(a) + ρ(b) > |a − b| las bolas cubren el tramo y éste es libre.  Si alguna
    pareja no lo cumple (muestra pegada a un obstáculo) se recurre a la prueba
    exacta :func:`polyline_free`: el resultado es idéntico, pero la mayoría de
    los tramos se aceptan con una única consulta vectorizada.
    """
    HALF_DIAG = math.sqrt(0.5) + 1e-6

    def __init__(self, occ: np.ndarray):
        self.occ = np.asarray(occ, bool)
        self.dt = cv2.distanceTransform((~self.occ).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.h, self.w = self.occ.shape
        self.fast = self.exact = 0          # estadística de uso

    def accepts(self, pts) -> bool:
        """Aceptación rápida (condición suficiente de ruta libre)."""
        P = densify(np.asarray(pts, dtype=float).reshape(-1, 2), 0.5)
        c = np.rint(P[:, 0]).astype(np.int64)
        r = np.rint(P[:, 1]).astype(np.int64)
        if c.min() < 0 or r.min() < 0 or c.max() >= self.w or r.max() >= self.h:
            return False
        rho = self.dt[r, c] - np.hypot(P[:, 0] - c, P[:, 1] - r) - self.HALF_DIAG
        if len(P) == 1:
            return bool(rho[0] > 0)
        gap = np.hypot(*np.diff(P, axis=0).T)
        return bool(np.all(rho[:-1] + rho[1:] > gap))

    def free(self, pts) -> bool:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return True
        if self.accepts(pts):
            self.fast += 1
            return True
        self.exact += 1
        return polyline_free(self.occ, pts)

    def segment(self, p, q) -> bool:
        return self.free(np.vstack([np.asarray(p, float), np.asarray(q, float)]))


def segment_free(occ: np.ndarray, p, q, step: float | None = None) -> bool:
    """True si el segmento p->q no atraviesa obstáculos (prueba exacta)."""
    return polyline_free(occ, np.vstack([np.asarray(p, float), np.asarray(q, float)]))


def densify(pts: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Remuestrea una polilínea con puntos cada ``step`` píxeles como máximo."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts.copy()
    d = np.diff(pts, axis=0)
    n = np.maximum(1, np.ceil(np.hypot(d[:, 0], d[:, 1]) / step).astype(np.int64))
    idx = np.repeat(np.arange(len(d)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(idx.size) - start + 1) / np.repeat(n, n)
    return np.vstack([pts[:1], pts[idx] + frac[:, None] * d[idx]])


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def turning_angles(pts: np.ndarray) -> np.ndarray:
    """Ángulo de giro (rad, 0..pi) en cada vértice interior de la polilínea."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return np.zeros(0)
    d1 = pts[1:-1] - pts[:-2]
    d2 = pts[2:] - pts[1:-1]
    n1 = np.linalg.norm(d1, axis=1)
    n2 = np.linalg.norm(d2, axis=1)
    ok = (n1 > 1e-12) & (n2 > 1e-12)
    cosang = np.ones(len(d1))
    cosang[ok] = np.einsum("ij,ij->i", d1[ok], d2[ok]) / (n1[ok] * n2[ok])
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def curvature(pts: np.ndarray) -> np.ndarray:
    """Curvatura discreta (1/px) por el círculo que pasa por 3 puntos."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return np.zeros(0)
    a = pts[:-2]
    b = pts[1:-1]
    c = pts[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    denom = ab * bc * ca
    k = np.zeros(len(a))
    ok = denom > 1e-12
    k[ok] = 2.0 * cross[ok] / denom[ok]
    return k


def bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Interpolación bilineal de ``img`` en puntos ``(x, y)``."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    h, w = img.shape
    x = np.clip(pts[:, 0], 0, w - 1)
    y = np.clip(pts[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
         + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return v


def ray_free_length(occ: np.ndarray, origin, direction, max_len: float, step: float = 0.5) -> float:
    """Distancia libre desde ``origin`` en ``direction`` hasta el primer obstáculo."""
    d = np.asarray(direction, dtype=float)
    nrm = np.hypot(*d)
    if nrm < 1e-12:
        return 0.0
    d = d / nrm
    n = max(2, int(np.ceil(max_len / step)) + 1)
    t = np.linspace(0, max_len, n)
    pts = np.asarray(origin, dtype=float)[None, :] + t[:, None] * d[None, :]
    hit = points_in_collision(occ, pts)
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return float(max_len)
    return float(t[max(idx[0] - 1, 0)])


def point_segment_distance(pts: np.ndarray, a, b) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return np.hypot(*(pts - a).T)
    t = np.clip(((pts - a) @ ab) / L2, 0, 1)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.hypot(*(pts - proj).T)


def simplify_collision_free(occ: np.ndarray, pts: np.ndarray, keep_first_last: bool = True,
                            step: float = 0.25) -> np.ndarray:
    """Subdivisión tipo Douglas-Peucker guiada por colisión.

    Devuelve el mínimo subconjunto de vértices (siempre incluyendo extremos)
    tal que cada cuerda entre vértices consecutivos está libre de colisión.
    Si dos puntos consecutivos de la polilínea original no tienen cuerda libre
    (p.ej. esqueleto pegado a un obstáculo en un paso estrecho) se conservan.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n <= 2:
        return pts.copy()
    if isinstance(occ, FastChecker):          # acepta también un FastChecker
        seg = occ.segment
    else:
        def seg(p, q):
            return segment_free(occ, p, q, step)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i <= 1:
            continue
        if seg(pts[i], pts[j]):
            continue
        d = point_segment_distance(pts[i + 1:j], pts[i], pts[j])
        k = i + 1 + int(np.argmax(d))
        keep[k] = True
        stack.append((i, k))
        stack.append((k, j))
    return pts[keep]
