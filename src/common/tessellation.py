"""Etapa 2 del teselado: expansión simultánea por frentes de onda cuadrados.

Equivalente vectorizado del ciclo de dilatación de ``Teselado.m``:

* cada obstáculo (componente conexa del espacio de configuración, incluido el
  marco) es un generador con su propia etiqueta/tonalidad;
* en cada iteración ``p`` todos los píxeles libres adyacentes (8-vecindad =
  frente de onda cuadrado) a una tesela se incorporan a ella con peso ``p``;
* si dos frentes distintos alcanzan un píxel en la misma iteración, ese píxel
  es un punto de colisión (parte de la frontera/esqueleto).  Se asigna a la
  etiqueta menor para que el teselado sea una partición completa.

El resultado es la partición en teselas (diagrama de Voronoi discreto con
métrica de Chebyshev) y la matriz de pesos ``M_p``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tessellation:
    labels: np.ndarray        # int32 (H, W) etiqueta de tesela de cada píxel (1..n)
    n_tiles: int              # número de teselas (= generadores)
    wave: np.ndarray          # int32 (H, W) peso p: nº de dilataciones para llegar (0 en obstáculos)
    collision: np.ndarray     # bool (H, W) píxeles donde chocaron frentes distintos
    clearance: np.ndarray     # float32 (H, W) distancia euclidiana al obstáculo más cercano [px]
    occ: np.ndarray           # bool (H, W) espacio de configuración ocupado
    iterations: int           # nº total de dilataciones realizadas
    comp_of_label: np.ndarray  # obstáculo (componente conexa) al que pertenece cada etiqueta
    mode: str = "obstacles"

    @property
    def free(self) -> np.ndarray:
        return ~self.occ


_OFFS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def generator_labels(occ: np.ndarray, mode: str = "obstacles", segment_px: float = 0.0):
    """Etiquetas de los generadores de frentes de onda.

    ``"obstacles"`` (artículo): cada obstáculo conexo es un generador.
    ``"segments"``: además, los obstáculos cuyo tamaño supera ``segment_px``
    (paredes, marco) se dividen en tramos de ese tamaño.  Así aparecen
    fronteras en pasillos y habitaciones delimitados por un único obstáculo.
    Devuelve (etiquetas, nº etiquetas, comp_of_label, frontera_mask).
    """
    n_lab, comp = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
    comp = comp.astype(np.int32)
    free_u8 = (~occ).astype(np.uint8)
    boundary = cv2.dilate(free_u8, np.ones((3, 3), np.uint8)).astype(bool) & occ
    if mode == "obstacles" or segment_px <= 0:
        return comp, n_lab - 1, np.arange(n_lab, dtype=np.int32), boundary
    if mode != "segments":
        raise ValueError("mode debe ser 'obstacles' o 'segments'")
    s = float(segment_px)
    ys, xs = np.nonzero(boundary)
    cid = comp[ys, xs].astype(np.int64)
    # extensión de cada obstáculo: solo se dividen los mayores que s
    xmin = np.full(n_lab, np.iinfo(np.int64).max)
    xmax = np.full(n_lab, -1)
    ymin = xmin.copy()
    ymax = xmax.copy()
    np.minimum.at(xmin, cid, xs)
    np.maximum.at(xmax, cid, xs)
    np.minimum.at(ymin, cid, ys)
    np.maximum.at(ymax, cid, ys)
    big = np.maximum(xmax - xmin, ymax - ymin) > s
    ncell = int(np.ceil(max(occ.shape) / s)) + 1
    cell = (ys // s).astype(np.int64) * ncell + (xs // s).astype(np.int64)
    key = cid * (ncell * ncell + 1) + np.where(big[cid], cell + 1, 0)
    uniq, inv = np.unique(key, return_inverse=True)
    new = (inv + 1).astype(np.int32)
    comp_of = np.concatenate([[0], uniq // (ncell * ncell + 1)]).astype(np.int32)
    labels = np.zeros_like(comp)
    # Interior de los obstáculos: una etiqueta cualquiera de su componente.
    # No se propaga (no está en el frente) y las grietas entre píxeles de un
    # mismo obstáculo se ignoran al extraer el esqueleto.
    first_lab = np.ones(n_lab, dtype=np.int32)
    first_lab[cid] = new
    labels[occ] = first_lab[comp[occ]]
    labels[ys, xs] = new
    return labels, len(uniq), comp_of, boundary


def tessellate(occ: np.ndarray, mode: str = "obstacles", segment_px: float = 0.0) -> Tessellation:
    occ = np.asarray(occ, dtype=bool)
    h, w = occ.shape
    comp, n_tiles, comp_of, near_free = generator_labels(occ, mode, segment_px)

    hp, wp = h + 2, w + 2
    lab = np.full((hp, wp), -1, dtype=np.int32)       # -1 = fuera del mapa
    lab[1:-1, 1:-1] = comp                             # 0 = libre sin asignar
    flat = lab.ravel()
    wave = np.zeros(hp * wp, dtype=np.int32)
    coll = np.zeros(hp * wp, dtype=bool)
    offs = np.array([dy * wp + dx for dy, dx in _OFFS8], dtype=np.int64)

    # Frente inicial: píxeles de obstáculo con al menos un vecino libre
    free_u8 = (~occ).astype(np.uint8)
    ys, xs = np.nonzero(near_free)
    front = ((ys + 1) * wp + (xs + 1)).astype(np.int64)

    k = 0
    while front.size:
        nb = (front[:, None] + offs[None, :]).ravel()
        src = np.repeat(flat[front], 8)
        m = flat[nb] == 0
        if not m.any():
            break
        k += 1
        nb = nb[m]
        src = src[m]
        order = np.lexsort((src, nb))
        nb = nb[order]
        src = src[order]
        uniq, first, counts = np.unique(nb, return_index=True, return_counts=True)
        minlab = src[first]
        maxlab = src[first + counts - 1]
        flat[uniq] = minlab
        wave[uniq] = k
        coll[uniq] = maxlab != minlab
        front = uniq

    labels = flat.reshape(hp, wp)[1:-1, 1:-1].copy()
    wave_img = wave.reshape(hp, wp)[1:-1, 1:-1].copy()
    coll_img = coll.reshape(hp, wp)[1:-1, 1:-1].copy()
    clearance = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5).astype(np.float32)
    return Tessellation(labels=labels, n_tiles=int(n_tiles), wave=wave_img, collision=coll_img,
                        clearance=clearance, occ=occ.copy(), iterations=k, comp_of_label=comp_of,
                        mode=mode)
