"""Entorno de navegación: carga de imagen, segmentación (M_b) y escala.

Convención de coordenadas usada en todo el paquete
--------------------------------------------------
* Las matrices se indexan ``[fila, columna]``.
* Los puntos se representan como ``(x, y) = (columna, fila)`` en píxeles,
  con el centro de cada píxel en coordenadas enteras.  Así se pueden graficar
  directamente con ``matplotlib.imshow`` (origen arriba a la izquierda).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Environment:
    """Mapa binario de obstáculos + escala física.

    ``obstacles`` es la matriz binaria M_b del artículo (True = obstáculo).
    """

    obstacles: np.ndarray
    meters_per_pixel: float = 20.0 / 480.0
    name: str = "entorno"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.obstacles = np.ascontiguousarray(self.obstacles, dtype=bool)
        if self.obstacles.ndim != 2:
            raise ValueError("El mapa de obstáculos debe ser una matriz 2D")
        if self.obstacles.shape[0] < 8 or self.obstacles.shape[1] < 8:
            raise ValueError("El mapa es demasiado pequeño (mínimo 8x8 píxeles)")
        if self.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel debe ser positivo")

    # ------------------------------------------------------------------ props
    @property
    def shape(self) -> tuple[int, int]:
        return self.obstacles.shape  # (filas, columnas)

    @property
    def height(self) -> int:
        return self.obstacles.shape[0]

    @property
    def width(self) -> int:
        return self.obstacles.shape[1]

    @property
    def width_m(self) -> float:
        return self.width * self.meters_per_pixel

    @property
    def height_m(self) -> float:
        return self.height * self.meters_per_pixel

    # ------------------------------------------------------------ conversions
    def m_to_px(self, value_m: float) -> float:
        return value_m / self.meters_per_pixel

    def px_to_m(self, value_px):
        return np.asarray(value_px, dtype=float) * self.meters_per_pixel

    def is_free(self, point_xy) -> bool:
        x, y = int(round(point_xy[0])), int(round(point_xy[1]))
        if not (0 <= y < self.height and 0 <= x < self.width):
            return False
        return not self.obstacles[y, x]

    # -------------------------------------------------------------- builders
    @classmethod
    def from_image(
        cls,
        path: str | Path,
        width_m: float = 20.0,
        obstacles: str = "auto",
        threshold: int | None = None,
    ) -> "Environment":
        """Carga una imagen (png, jpg, bmp, pgm...) y la segmenta.

        Parameters
        ----------
        width_m:
            Ancho físico del entorno representado por la imagen [m].
        obstacles:
            ``"white"`` (convención del artículo), ``"black"`` (mapas tipo
            ROS/occupancy grid) o ``"auto"`` (la clase minoritaria se asume
            como obstáculo).
        threshold:
            Umbral de binarización 0-255. ``None`` usa Otsu.
        """
        path = Path(path)
        data = np.fromfile(str(path), dtype=np.uint8)  # soporta rutas con tildes
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        gray = to_gray(img)
        mask = segment(gray, obstacles=obstacles, threshold=threshold)
        mpp = width_m / mask.shape[1]
        return cls(mask, mpp, name=path.stem, meta={"source": str(path)})

    @classmethod
    def from_array(cls, array, width_m: float = 20.0, name: str = "entorno",
                   obstacles: str = "white", threshold: int | None = 127) -> "Environment":
        arr = np.asarray(array)
        if arr.dtype == bool:
            mask = arr
        else:
            mask = segment(to_gray(arr), obstacles=obstacles, threshold=threshold)
        return cls(mask, width_m / mask.shape[1], name=name)

    def save_image(self, path: str | Path) -> None:
        img = np.where(self.obstacles, 255, 0).astype(np.uint8)
        ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
        if not ok:
            raise IOError(f"No se pudo codificar la imagen {path}")
        buf.tofile(str(path))


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convierte cualquier imagen (gris, RGB, RGBA, 16 bits) a gris uint8."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        rng = img.max() - img.min()
        img = ((img - img.min()) / (rng if rng > 0 else 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        # Componer sobre fondo negro: zonas transparentes = espacio libre
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[:, :, 0]


def segment(gray: np.ndarray, obstacles: str = "auto", threshold: int | None = None) -> np.ndarray:
    """Segmentación de la imagen -> matriz binaria M_b (True = obstáculo)."""
    if threshold is None:
        if gray.min() == gray.max():
            white = np.zeros_like(gray, dtype=bool) if gray.max() < 128 else np.ones_like(gray, dtype=bool)
        else:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white = bw > 0
    else:
        white = gray > threshold
    if obstacles == "white":
        return white
    if obstacles == "black":
        return ~white
    if obstacles == "auto":
        # En casi todos los mapas el espacio libre es mayoritario
        return white if white.mean() <= 0.5 else ~white
    raise ValueError("obstacles debe ser 'white', 'black' o 'auto'")
