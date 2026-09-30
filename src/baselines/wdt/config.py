"""Parámetros del planificador.

Todas las distancias físicas están en metros; el planificador las convierte a
píxeles usando la escala del entorno (``Environment.meters_per_pixel``).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PlannerConfig:
    # --- Robot -------------------------------------------------------------
    robot_diameter: float = 0.5          # D en el artículo [m]
    safety_margin: float = 0.0           # holgura extra sobre D/2 [m]
    dilation_shape: str = "disk"         # "disk" (círculo circunscrito) o "square"

    # --- Teselado / esqueleto ---------------------------------------------
    generators: str = "obstacles"        # "obstacles" (artículo) o "segments" (paredes divididas)
    segment_size: float = 2.0            # [m] tamaño de tramo en modo "segments"
    node_merge_dist: float = 3.0         # [px] aristas más cortas que esto se contraen

    # --- Generación de ruta -----------------------------------------------
    route_method: str = "paper"          # "paper" (voraz + n-1 alternativas) o "dijkstra"
    optimize: bool = True                # mejoras propias: elegir la ruta más corta DESPUÉS de
                                         # reducir (incluye Dijkstra como candidata) + tensado final.
                                         # False = solo el método del artículo.

    # --- Reducción --------------------------------------------------------
    reduction_min_dist: float | None = None  # [px]; None => D en píxeles (artículo)
    reduction_max_cycles: int = 60

    # --- Suavizado --------------------------------------------------------
    smoothing: str = "piecewise"         # "piecewise", "global" (Bézier único) o "none"
    smoothing_tension: float = 1.0       # 0..1, magnitud relativa de las tangentes
    smoothing_weight_gain: float = 1.0   # factor sobre el peso transversal
    smoothing_samples_per_px: float = 1.0

    # --- Varios -----------------------------------------------------------
    collision_step: float = 0.25         # [px] paso de muestreo para verificar colisión

    def to_dict(self) -> dict:
        return asdict(self)
