"""Etapa 1 del teselado: dilatación inicial de obstáculos (D/2).

Tras esta etapa el robot puede tratarse como un punto (artículo, sección 2.1,
paso 3).  Los bordes del entorno se tratan como un obstáculo más (el "marco"),
igual que en ``Teselado.m``, pero también se dilatan para que el robot no roce
las paredes.
"""
from __future__ import annotations

import cv2
import numpy as np


def robot_radius_px(robot_diameter_m: float, safety_margin_m: float, meters_per_pixel: float) -> int:
    r = (robot_diameter_m / 2.0 + safety_margin_m) / meters_per_pixel
    return max(0, int(np.ceil(r - 1e-9)))


def add_frame(obstacles: np.ndarray, thickness: int = 1) -> np.ndarray:
    occ = obstacles.copy()
    t = max(1, int(thickness))
    occ[:t, :] = True
    occ[-t:, :] = True
    occ[:, :t] = True
    occ[:, -t:] = True
    return occ


def dilate(obstacles: np.ndarray, radius_px: int, shape: str = "disk") -> np.ndarray:
    """Dilata los obstáculos ``radius_px`` píxeles (círculo o cuadrado)."""
    if radius_px <= 0:
        return obstacles.copy()
    k = 2 * radius_px + 1
    if shape == "disk":
        yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        kernel = ((xx * xx + yy * yy) <= radius_px * radius_px + 1e-9).astype(np.uint8)
    elif shape == "square":
        kernel = np.ones((k, k), np.uint8)
    else:
        raise ValueError("shape debe ser 'disk' o 'square'")
    out = cv2.dilate(obstacles.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return out.astype(bool)


def configuration_space(obstacles: np.ndarray, radius_px: int, shape: str = "disk") -> np.ndarray:
    """Devuelve el espacio de configuración ocupado (marco incluido)."""
    return dilate(add_frame(obstacles), radius_px, shape)
