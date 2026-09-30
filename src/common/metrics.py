"""Métricas comunes para comparar cualquier ruta en igualdad de condiciones.

Criterios
---------
* Validez: la ruta no atraviesa el espacio de configuración (prueba exacta).
* Longitud [m] y relación con la ruta óptima de referencia.
* Costo computacional: tiempo [ms] y nodos/estados/muestras procesados.
* Seguridad: holgura mínima y media hasta los obstáculos REALES.
* Suavidad: giro total, número de giros bruscos (> 45°), radio de giro mínimo.
* Cinemática:
  - holonómico (omnidireccional): puede seguir cualquier ruta sin colisión;
    tiempo = L / v.
  - diferencial: puede girar sobre sí mismo, pero en cada esquina cerrada
    (radio < radio del robot) debe detenerse y rotar; tiempo = L/v + Σθ/ω.
  - Ackermann (auto): no puede girar en el sitio; la ruta es factible sólo si
    el radio de giro mínimo alcanzable ≥ R_min.

El radio de giro de la ruta se mide con la curvatura real: la ruta se
remuestrea cada Δ = 0.25 m y κ ≈ Δθ/Δ.  Un quiebre de rumbo θ cuenta como
curvatura θ/Δ: con R_min = 1 m se toleran quiebres de hasta ~14°, que un auto
redondea desviándose menos de ~1-2 cm; las esquinas de 45° de una malla no.  Las esquinas de una poligonal son
discontinuidades de rumbo (κ → ∞, un auto no puede seguirlas sin maniobrar);
las curvas suaves (Bézier, arcos de Hybrid A*) conservan su curvatura.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from common.geometry import bilinear, densify, polyline_free, polyline_length
from common.problem import AlgorithmResult, Problem


def simplify(path: np.ndarray, eps: float = 0.75) -> np.ndarray:
    if len(path) < 3:
        return np.asarray(path, float)
    a = cv2.approxPolyDP(np.asarray(path, np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2)
    return a.astype(float)


def resample(path: np.ndarray, delta: float) -> np.ndarray:
    """Remuestreo por longitud de arco con paso ``delta`` (incluye los extremos)."""
    P = np.asarray(path, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return P[:1]
    t = np.arange(0.0, s[-1], delta)
    t = np.append(t, s[-1]) if s[-1] - t[-1] > 1e-6 else t
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], axis=1)


def corner_radii(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ángulos de giro [rad], radios alcanzables [px]) en cada vértice interior."""
    P = np.asarray(path, float)
    if len(P) < 3:
        return np.zeros(0), np.zeros(0)
    d1 = P[1:-1] - P[:-2]
    d2 = P[2:] - P[1:-1]
    l1 = np.linalg.norm(d1, axis=1)
    l2 = np.linalg.norm(d2, axis=1)
    ok = (l1 > 1e-9) & (l2 > 1e-9)
    cos = np.ones(len(d1))
    cos[ok] = np.einsum("ij,ij->i", d1[ok], d2[ok]) / (l1[ok] * l2[ok])
    th = np.arccos(np.clip(cos, -1, 1))
    r = np.full(len(th), np.inf)
    m = th > 1e-4
    r[m] = 0.5 * np.minimum(l1[m], l2[m]) / np.tan(th[m] / 2)
    return th, r


def evaluate(problem: Problem, res: AlgorithmResult, ref_length_px: float | None = None,
             v: float = 1.0, omega_deg: float = 90.0, R_min_m: float = 1.0) -> dict:
    mpp = problem.mpp
    m = {"exito": bool(res.success), "tiempo_ms": res.time_s * 1000.0}
    effort = None
    for k in ("nodos_expandidos", "estados_expandidos", "iteraciones", "muestras", "pasos", "vertices"):
        if k in res.stats:
            effort = res.stats[k]
            m["esfuerzo"] = f"{effort} {k.replace('_', ' ')}"
            m["esfuerzo_n"] = float(effort)
            break
    if not res.success or res.path is None or len(res.path) < 2:
        res.metrics = m
        return m
    P = np.asarray(res.path, float)
    L = polyline_length(P)
    dense = densify(P, 0.5)
    real = bilinear(problem.real_clearance, dense)
    S = simplify(P)
    th, _ = corner_radii(S)
    robot_r = problem.radius_px
    sharp = th > math.radians(45)
    # Curvatura real: se remuestrea la ruta cada Δ y se mide el giro entre tramos
    # consecutivos (κ ≈ Δθ/Δ).  Una esquina de una poligonal es una discontinuidad
    # de rumbo (κ enorme); una curva suave conserva su curvatura verdadera.
    # Δ a escala del vehículo (0.25 m): con Δ de pocos píxeles, errores de 0.01 px en la
    # discretización de un arco ya dan ~7 % de error en κ.
    delta = max(2.0, 0.25 / mpp)
    Rs = resample(P, delta)
    tk, _ = corner_radii(Rs)
    if len(tk):
        kappa = tk / delta
        rmin = float(1.0 / kappa.max()) if kappa.max() > 1e-9 else math.inf
        # rotaciones en el sitio de un robot diferencial: donde el radio local < su radio
        stop_turns = float(tk[(delta / np.maximum(tk, 1e-12)) < robot_r].sum())
    else:
        rmin, stop_turns = math.inf, 0.0
    m.update({
        "sin_colision": bool(polyline_free(problem.occ, P)),
        "longitud_m": L * mpp,
        "relacion_optima": (L / ref_length_px) if ref_length_px else np.nan,
        "holgura_min_m": float(real.min()) * mpp,
        "holgura_media_m": float(real.mean()) * mpp,
        "giro_total_deg": float(np.degrees(th.sum())) if len(th) else 0.0,
        "giros_bruscos": int(sharp.sum()),
        "vertices": int(len(S)),
        "radio_giro_min_m": rmin * mpp if math.isfinite(rmin) else float("inf"),
        "tiempo_holonomico_s": L * mpp / v,
        "tiempo_diferencial_s": L * mpp / v + math.degrees(stop_turns) / omega_deg,
        "factible_ackermann": bool(rmin * mpp >= 0.97 * R_min_m),   # 3 %: error de discretizar κ
    })
    m["tiempo_ackermann_s"] = m["tiempo_holonomico_s"] if m["factible_ackermann"] else float("inf")
    res.metrics = m
    return m


CRITERIA = [
    # clave, etiqueta, mejor ("min"/"max"/None), formato, descripción
    ("exito", "Éxito", "max", "{}", "¿Encontró una ruta de S a T?"),
    ("sin_colision", "Sin colisión", "max", "{}", "La ruta no atraviesa obstáculos dilatados (prueba exacta)."),
    ("longitud_m", "Longitud [m]", "min", "{:.2f}", "Longitud de la ruta."),
    ("relacion_optima", "L / L óptima", "min", "{:.3f}", "Longitud dividida por la del grafo de visibilidad "
     "(óptimo euclidiano). 1.000 = óptima."),
    ("tiempo_ms", "Tiempo de cómputo [ms]", "min", "{:.1f}", "Tiempo de CPU para calcular la ruta."),
    ("esfuerzo_n", "Esfuerzo (nodos/muestras)", None, "{:.0f}", "Nodos expandidos, estados, iteraciones o muestras "
     "(unidades distintas según el algoritmo: informativo, no se compara)."),
    ("holgura_min_m", "Holgura mínima [m]", "max", "{:.3f}", "Distancia mínima del centro del robot a un "
     "obstáculo real (≥ radio del robot = no toca)."),
    ("holgura_media_m", "Holgura media [m]", "max", "{:.2f}", "Distancia media a los obstáculos: seguridad global."),
    ("giro_total_deg", "Giro total [°]", "min", "{:.0f}", "Suma de los ángulos de giro (esfuerzo de dirección)."),
    ("giros_bruscos", "Giros bruscos (>45°)", "min", "{}", "Esquinas cerradas que obligan a frenar."),
    ("radio_giro_min_m", "Radio de giro mín. [m]", "max", "{:.2f}", "Radio del giro más cerrado; un auto necesita "
     "que sea ≥ su radio de giro mínimo."),
    ("tiempo_holonomico_s", "Recorrido holonómico [s]", "min", "{:.1f}", "Tiempo de recorrido de un robot "
     "omnidireccional (L / v)."),
    ("tiempo_diferencial_s", "Recorrido diferencial [s]", "min", "{:.1f}", "Robot diferencial: L/v más el tiempo "
     "de rotar en el sitio en las esquinas cerradas."),
    ("factible_ackermann", "Factible Ackermann", "max", "{}", "¿Un vehículo tipo auto puede seguirla sin maniobras?"),
]
