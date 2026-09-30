"""Infraestructura común a todos los algoritmos de planificación.

Todos reciben el MISMO ``Problem``: el espacio de configuración (obstáculos
dilatados por el radio del robot), S y T, y la escala.  Así la comparación es
justa: la ruta de cualquier algoritmo se valida con la misma prueba exacta de
colisión y se mide con las mismas métricas (``metrics.py``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from common.cspace import configuration_space, robot_radius_px
from common.environment import Environment


@dataclass
class Problem:
    env: Environment
    occ: np.ndarray                  # espacio de configuración ocupado (bool)
    start: np.ndarray                # (x, y) px, en espacio libre
    goal: np.ndarray
    robot_diameter: float            # [m]
    safety_margin: float             # [m]
    radius_px: int
    seed: int = 0
    warnings: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def mpp(self) -> float:
        return self.env.meters_per_pixel

    @property
    def shape(self):
        return self.occ.shape

    @property
    def clearance(self) -> np.ndarray:
        """Distancia euclidiana [px] al obstáculo del espacio de configuración."""
        if "clearance" not in self._cache:
            self._cache["clearance"] = cv2.distanceTransform((~self.occ).astype(np.uint8), cv2.DIST_L2, 5)
        return self._cache["clearance"]

    @property
    def real_clearance(self) -> np.ndarray:
        """Distancia [px] del centro del robot al obstáculo REAL más cercano."""
        if "real_clearance" not in self._cache:
            free = (~self.env.obstacles).astype(np.uint8)
            free[[0, -1], :] = 0
            free[:, [0, -1]] = 0
            self._cache["real_clearance"] = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        return self._cache["real_clearance"]

    @property
    def free_comp(self) -> np.ndarray:
        if "free_comp" not in self._cache:
            _, comp = cv2.connectedComponents((~self.occ).astype(np.uint8), connectivity=8)
            self._cache["free_comp"] = comp
        return self._cache["free_comp"]

    def connected(self) -> bool:
        c = self.free_comp
        return c[int(round(self.start[1])), int(round(self.start[0]))] == \
            c[int(round(self.goal[1])), int(round(self.goal[0]))]

    @classmethod
    def build(cls, env: Environment, start, goal, robot_diameter: float = 0.5, safety_margin: float = 0.0,
              shape: str = "disk", seed: int = 0, occ: np.ndarray | None = None) -> "Problem":
        r = robot_radius_px(robot_diameter, safety_margin, env.meters_per_pixel)
        occ = configuration_space(env.obstacles, r, shape) if occ is None else occ
        p = cls(env, occ, np.zeros(2), np.zeros(2), robot_diameter, safety_margin, r, seed)
        p.start, ds = snap_free(occ, start)
        p.goal, dg = snap_free(occ, goal)
        if ds > 0:
            p.warnings.append(f"S estaba dentro de un obstáculo dilatado; se movió {ds * env.meters_per_pixel:.2f} m")
        if dg > 0:
            p.warnings.append(f"T estaba dentro de un obstáculo dilatado; se movió {dg * env.meters_per_pixel:.2f} m")
        return p


def snap_free(occ: np.ndarray, p) -> tuple[np.ndarray, float]:
    p = np.asarray(p, dtype=float)
    h, w = occ.shape
    r = min(max(int(round(p[1])), 0), h - 1)
    c = min(max(int(round(p[0])), 0), w - 1)
    if not occ[r, c]:
        return np.array([c, r], float), 0.0
    ys, xs = np.nonzero(~occ)
    if len(ys) == 0:
        raise ValueError("No hay espacio libre en el mapa")
    d = (xs - p[0]) ** 2 + (ys - p[1]) ** 2
    k = int(np.argmin(d))
    return np.array([xs[k], ys[k]], float), float(np.sqrt(d[k]))


@dataclass
class AlgorithmResult:
    key: str
    name: str
    success: bool
    path: np.ndarray | None
    time_s: float = 0.0
    message: str = ""
    stats: dict = field(default_factory=dict)      # nodos expandidos, iteraciones, muestras...
    viz: dict = field(default_factory=dict)        # datos para dibujar (árbol, visitados, feromona...)
    metrics: dict = field(default_factory=dict)


@dataclass
class ParamSpec:
    name: str
    label: str
    default: Any
    kind: str = "float"              # float | int | bool | choice
    minimum: float = 0.0
    maximum: float = 1e6
    step: float = 1.0
    choices: tuple = ()
    tip: str = ""
    decimals: int = 2
    suffix: str = ""


@dataclass
class AlgorithmSpec:
    key: str
    name: str
    icon: str
    family: str
    summary: str                     # una línea
    description: str                 # explicación (HTML sencillo)
    properties: dict                 # completitud, optimalidad, determinismo, mapa, cinemática
    params: list[ParamSpec]
    run: Callable[..., AlgorithmResult]
    auto_run: bool = True            # rápido: se puede re-ejecutar al mover S/T

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def execute(self, problem: Problem, **params) -> AlgorithmResult:
        kw = self.defaults()
        kw.update(params)
        t0 = time.perf_counter()
        if not problem.connected():
            res = AlgorithmResult(self.key, self.name, False, None,
                                  message="No existe ruta: S y T están en regiones libres desconectadas")
        else:
            try:
                res = self.run(problem, **kw)
            except Exception as exc:  # noqa: BLE001
                import traceback
                res = AlgorithmResult(self.key, self.name, False, None,
                                      message=f"Error: {exc}", stats={"traceback": traceback.format_exc()})
        res.key, res.name = self.key, self.name
        if res.time_s == 0.0:
            res.time_s = time.perf_counter() - t0
        return res


def coarse_grid(occ: np.ndarray, cell: int) -> np.ndarray:
    """Malla gruesa conservadora: una celda está ocupada si CUALQUIER píxel lo está."""
    cell = max(1, int(cell))
    if cell == 1:
        return occ.copy()
    h, w = occ.shape
    H, W = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    pad = np.ones((H * cell, W * cell), bool)
    pad[:h, :w] = occ
    return pad.reshape(H, cell, W, cell).any(axis=(1, 3))


def cell_center(r, c, cell: int):
    """Centro (x, y) en píxeles de la celda (r, c)."""
    return np.array([c * cell + (cell - 1) / 2.0, r * cell + (cell - 1) / 2.0])


def point_cell(p, cell: int, shape) -> tuple[int, int]:
    H, W = shape
    r = int(np.clip(np.floor((p[1] + 0.5) / cell), 0, H - 1))
    c = int(np.clip(np.floor((p[0] + 0.5) / cell), 0, W - 1))
    return r, c
