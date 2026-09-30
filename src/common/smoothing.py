"""Suavizado de la ruta reducida con curvas de Bézier (artículo, sección 2.3).

    B(u) = Σ_{i=0..n} J_{n,i}(u) P_i ,   J_{n,i}(u) = C(n,i) u^i (1-u)^(n-i)

El artículo indica que los puntos de control se ubican sobre líneas
*transversales* a los nodos de la ruta reducida y que su distancia al nodo
corresponde al peso del mapa M_p.  Se implementan dos variantes:

``piecewise`` (por defecto, robusta)
    Bézier cúbicas encadenadas con continuidad C1 que pasan por los nodos de
    la ruta reducida (via points, ref. [5] del artículo).  En cada nodo la
    tangente es la bisectriz del giro y la longitud de los manejadores se
    limita con el **peso transversal**: la distancia libre medida sobre la
    línea transversal hacia el lado exterior del giro.  Luego se verifica
    colisión: si un tramo colisiona se inserta un via point en su punto medio
    (elimina inflexiones) y/o se reduce la tensión de sus extremos.  Con
    tensión cero el tramo coincide con la recta de la ruta reducida, que ya
    es libre, por lo que el procedimiento siempre termina con una curva libre
    de colisión.

``global``
    Una única curva de Bézier de grado n (ecuación (1) del artículo).  Puntos
    de control: S, los nodos desplazados sobre su transversal exterior una
    fracción del peso, y T.  No garantiza ausencia de colisión; se reporta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.geometry import densify, polyline_free, polyline_length, ray_free_length


@dataclass
class SmoothResult:
    curve: np.ndarray
    method: str
    valid: bool
    control_polygons: list[np.ndarray] = field(default_factory=list)
    via_points: np.ndarray | None = None
    transverse: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    iterations: int = 0
    inserted: int = 0
    straightened: int = 0


def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.zeros(2)


def transverse_lines(occ, P: np.ndarray, max_len: float):
    """Líneas transversales (bisectriz) en cada nodo interior.

    Devuelve lista de (nodo, extremo_exterior, extremo_interior) y los pesos
    transversales exteriores/interiores (distancia libre en cada sentido).
    """
    lines = []
    w_out = np.full(len(P), max_len)
    w_in = np.full(len(P), max_len)
    for i in range(1, len(P) - 1):
        din = _unit(P[i] - P[i - 1])
        dout = _unit(P[i + 1] - P[i])
        inward = _unit(dout - din)
        if not inward.any():
            inward = np.array([-din[1], din[0]])
        outward = -inward
        wo = ray_free_length(occ, P[i], outward, max_len)
        wi = ray_free_length(occ, P[i], inward, max_len)
        w_out[i] = wo
        w_in[i] = wi
        lines.append((P[i].copy(), P[i] + outward * wo, P[i] + inward * wi))
    return lines, w_out, w_in


def cubic_points(ctrl: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1 - t
    return (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] + 3 * mt * (t ** 2) * ctrl[2] + (t ** 3) * ctrl[3]


def bezier_points(ctrl: np.ndarray, n: int) -> np.ndarray:
    """Evaluación de una Bézier de grado arbitrario (De Casteljau, estable)."""
    ctrl = np.asarray(ctrl, dtype=float)
    t = np.linspace(0.0, 1.0, n)[:, None, None]
    pts = np.broadcast_to(ctrl[None, :, :], (n,) + ctrl.shape).copy()
    for _ in range(len(ctrl) - 1):
        pts = (1 - t) * pts[:, :-1, :] + t * pts[:, 1:, :]
    return pts[:, 0, :]


def _samples_for(ctrl: np.ndarray, per_px: float) -> int:
    return max(8, int(np.ceil(polyline_length(ctrl) * per_px)) + 1)


def smooth_piecewise(occ, P: np.ndarray, tension: float = 1.0, weight_gain: float = 1.0,
                     samples_per_px: float = 1.0, max_iter: int = 80) -> SmoothResult:
    P = np.asarray(P, dtype=float)
    h, w = occ.shape
    max_len = float(max(h, w))
    lines, w_out, _ = transverse_lines(occ, P, max_len)
    if len(P) < 3:
        curve = densify(P, 1.0 / max(samples_per_px, 1e-3))
        return SmoothResult(curve, "piecewise", polyline_free(occ, P),
                            [np.vstack([P[0], P[0] * 2 / 3 + P[-1] / 3, P[0] / 3 + P[-1] * 2 / 3, P[-1]])],
                            P.copy(), lines)

    V = [p.copy() for p in P]
    wv = list(w_out)                    # peso transversal por via point
    inserted = [False] * len(V)
    tau = [float(tension)] * len(V)
    halvings = [0] * len(V)
    n_inserted = 0

    def handles():
        n = len(V)
        T = []
        A = []
        for i in range(n):
            if i == 0:
                t = _unit(V[1] - V[0])
                base = np.hypot(*(V[1] - V[0])) / 3.0
                cap = np.inf
            elif i == n - 1:
                t = _unit(V[-1] - V[-2])
                base = np.hypot(*(V[-1] - V[-2])) / 3.0
                cap = np.inf
            else:
                lin = np.hypot(*(V[i] - V[i - 1]))
                lout = np.hypot(*(V[i + 1] - V[i]))
                din = _unit(V[i] - V[i - 1])
                dout = _unit(V[i + 1] - V[i])
                t = _unit(din + dout)
                if not t.any():
                    t = dout
                base = 0.45 * min(lin, lout)
                theta = float(np.arccos(np.clip(din @ dout, -1, 1)))
                s = max(np.sin(theta / 2.0), 0.1)
                cap = weight_gain * wv[i] / s if theta > 1e-3 else np.inf
            T.append(t)
            A.append(tau[i] * min(base, cap))
        return T, A

    def segments():
        T, A = handles()
        segs = []
        for k in range(len(V) - 1):
            c = np.vstack([V[k], V[k] + A[k] * T[k], V[k + 1] - A[k + 1] * T[k + 1], V[k + 1]])
            segs.append(c)
        return segs

    def collides(c):
        # Tramo recto (tensión 0): exactamente la misma prueba que la reducción
        if np.allclose(c[1], c[0]) and np.allclose(c[2], c[3]):
            return not polyline_free(occ, c[[0, 3]])
        n = max(8, int(np.ceil(polyline_length(c) * 4)) + 1)   # ≤ 0.25 px entre muestras
        return not polyline_free(occ, cubic_points(c, n))

    it = 0
    for it in range(1, max_iter + 1):
        segs = segments()
        bad = [k for k, c in enumerate(segs) if collides(c)]
        if not bad:
            break
        # 1) insertar via point en tramos con inflexión entre dos nodos originales
        to_split = [k for k in bad if not inserted[k] and not inserted[k + 1]
                    and np.hypot(*(V[k + 1] - V[k])) > 6.0]
        for k in sorted(to_split, reverse=True):
            mid = 0.5 * (V[k] + V[k + 1])
            V.insert(k + 1, mid)
            wv.insert(k + 1, max_len)
            inserted.insert(k + 1, True)
            tau.insert(k + 1, float(tension))
            halvings.insert(k + 1, 0)
            n_inserted += 1
        if to_split:
            continue
        # 2) reducir tensión en los extremos de los tramos que colisionan
        for k in bad:
            for i in (k, k + 1):
                halvings[i] += 1
                tau[i] = 0.0 if halvings[i] > 6 else tau[i] * 0.5
    segs = segments()
    bad = [k for k, c in enumerate(segs) if collides(c)]
    straightened = 0
    while bad:                         # garantía final: tramo recto
        changed = False
        for k in bad:
            for i in (k, k + 1):
                if tau[i] != 0.0:
                    tau[i] = 0.0
                    straightened += 1
                    changed = True
        if not changed:                # ya son rectas: nada más que hacer
            break
        segs = segments()
        bad = [k for k, c in enumerate(segs) if collides(c)]

    curve = [cubic_points(c, _samples_for(c, samples_per_px)) for c in segs]
    curve = np.vstack([curve[0]] + [c[1:] for c in curve[1:]])
    valid = not any(collides(c) for c in segs)
    return SmoothResult(curve, "piecewise", valid, segs, np.asarray(V), lines, it, n_inserted, straightened)


def smooth_global(occ, P: np.ndarray, weight_gain: float = 1.0, samples_per_px: float = 1.0) -> SmoothResult:
    P = np.asarray(P, dtype=float)
    h, w = occ.shape
    lines, w_out, _ = transverse_lines(occ, P, float(max(h, w)))
    ctrl = [P[0]]
    for i in range(1, len(P) - 1):
        node, out_end, _ = lines[i - 1]
        direction = _unit(out_end - node)
        lin = np.hypot(*(P[i] - P[i - 1]))
        lout = np.hypot(*(P[i + 1] - P[i]))
        off = min(0.5 * weight_gain * w_out[i], 0.5 * min(lin, lout))
        ctrl.append(node + direction * off)
    ctrl.append(P[-1])
    ctrl = np.asarray(ctrl)
    n = max(50, int(np.ceil(polyline_length(ctrl) * max(samples_per_px, 1.0))))
    curve = bezier_points(ctrl, n)
    fine = bezier_points(ctrl, max(n, int(polyline_length(ctrl) * 4)))
    valid = polyline_free(occ, fine)
    return SmoothResult(curve, "global", valid, [ctrl], P.copy(), lines)


def smooth_none(occ, P: np.ndarray) -> SmoothResult:
    P = np.asarray(P, dtype=float)
    valid = polyline_free(occ, P)
    return SmoothResult(densify(P, 1.0), "none", valid, [], P.copy(), [])
