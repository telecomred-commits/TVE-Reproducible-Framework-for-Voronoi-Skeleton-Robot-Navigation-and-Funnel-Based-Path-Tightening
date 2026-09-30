"""Escenas de interiores y exteriores para las simulaciones del artículo.

Cada generador devuelve el mapa de obstáculos REAL (verdad de terreno) de un
recinto cuadrado de ``width_m`` metros, dibujado con dimensiones típicas de
mobiliario.  A partir de él se sintetiza la fotografía (``navplanner.synthetic``)
y el mapa que usa el robot es el que DETECTA el procesamiento de imagen.

Escenas: oficina, bodega, vivienda, taller, tienda, plaza exterior, hospital.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


class _Canvas:
    def __init__(self, size: int, width_m: float):
        self.n = size
        self.s = size / width_m                     # px por metro
        self.img = np.zeros((size, size), np.uint8)

    def px(self, v):
        return v * self.s

    def rect(self, cx, cy, w, h, ang=0.0):
        box = cv2.boxPoints(((self.px(cx), self.px(cy)), (self.px(w), self.px(h)), math.degrees(ang)))
        cv2.fillPoly(self.img, [np.round(box).astype(np.int32)], 1)

    def circle(self, cx, cy, r):
        cv2.circle(self.img, (int(round(self.px(cx))), int(round(self.px(cy)))), max(1, int(round(self.px(r)))), 1, -1)

    def ellipse(self, cx, cy, a, b, ang=0.0):
        cv2.ellipse(self.img, (int(round(self.px(cx))), int(round(self.px(cy)))),
                    (max(1, int(self.px(a))), max(1, int(self.px(b)))), math.degrees(ang), 0, 360, 1, -1)

    def poly(self, pts):
        cv2.fillPoly(self.img, [np.round(np.asarray(pts) * self.s).astype(np.int32)], 1)

    def ring(self, cx, cy, r_out, r_in, a0, a1):
        """Sector de corona (obstáculo cóncavo: banca curva, jardinera en U)."""
        t = np.linspace(a0, a1, 40)
        outer = np.stack([cx + r_out * np.cos(t), cy + r_out * np.sin(t)], 1)
        inner = np.stack([cx + r_in * np.cos(t[::-1]), cy + r_in * np.sin(t[::-1])], 1)
        self.poly(np.vstack([outer, inner]))

    def wall(self, x0, y0, x1, y1, th=0.15):
        cv2.line(self.img, (int(self.px(x0)), int(self.px(y0))), (int(self.px(x1)), int(self.px(y1))), 1,
                 max(1, int(round(self.px(th)))))

    @property
    def mask(self):
        return self.img.astype(bool)


def _free_rect(occupied, box, gap):
    x0, y0, x1, y1 = box
    return all(x1 + gap <= a or a1 + gap <= x0 or y1 + gap <= b or b1 + gap <= y0 for a, b, a1, b1 in occupied)


def office(seed=0, size=400, width_m=12.0):
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    # armarios y archivadores contra las paredes
    x = 0.4
    while x < W - 1.2:
        if rng.random() < 0.55:
            w = rng.uniform(0.8, 1.6)
            c.rect(x + w / 2, 0.3, w, 0.45)
        x += rng.uniform(1.2, 2.2)
    # islas de escritorios (2 x k) con sillas
    rows = rng.integers(2, 4)
    cols = rng.integers(2, 3)
    y = 1.8
    for r in range(rows):
        x = 1.3 + rng.uniform(0, 0.6)
        for q in range(cols):
            k = rng.integers(2, 4)
            dw, dh = 1.4, 0.75
            for i in range(k):
                for j in range(2):
                    cx = x + i * dw + dw / 2
                    cy = y + j * dh + dh / 2
                    c.rect(cx, cy, dw - 0.04, dh - 0.04)
                    if rng.random() < 0.8:
                        side = -1 if j == 0 else 1
                        c.circle(cx + rng.uniform(-0.3, 0.3), cy + side * (dh / 2 + 0.33), 0.24)
            x += k * dw + rng.uniform(1.4, 2.0)
            if x > W - 2:
                break
        y += 2 * dh + rng.uniform(1.6, 2.2)
        if y > W - 2.5:
            break
    # mesa de reuniones y plantas
    if y < W - 2.2:
        c.ellipse(W * rng.uniform(0.35, 0.65), min(W - 1.3, y + 0.6), 1.3, 0.6)
    for _ in range(rng.integers(2, 5)):
        c.circle(rng.uniform(0.6, W - 0.6), rng.uniform(0.6, W - 0.6), rng.uniform(0.18, 0.3))
    return c.mask


def warehouse(seed=0, size=400, width_m=16.0):
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    horizontal = rng.random() < 0.5
    depth = 0.9
    aisle = rng.uniform(1.5, 2.3)
    cross = rng.uniform(0.35, 0.6) * W
    pos = 1.6
    racks = []
    while pos < W - 3.5:
        L1 = cross - 1.2 - rng.uniform(0, 1.0)
        L2 = W - cross - 1.6 - rng.uniform(0, 1.5)
        for a, L in ((0.6, L1), (cross + 1.0, L2)):
            if L > 1.5:
                racks.append((a, pos, L))
        pos += 2 * depth + aisle
    for a, p, L in racks:
        for k in range(2):                               # estantería doble
            if horizontal:
                c.rect(a + L / 2, p + k * depth + depth / 2, L, depth - 0.05)
            else:
                c.rect(p + k * depth + depth / 2, a + L / 2, depth - 0.05, L)
    # columnas y estibas en la zona de despacho
    for i in range(1, 4):
        for j in range(1, 4):
            if rng.random() < 0.35:
                c.rect(i * W / 4, j * W / 4, 0.4, 0.4)
    zone = W - 3.0
    for _ in range(rng.integers(3, 7)):
        u = rng.uniform(0.8, W - 0.8)
        v = rng.uniform(zone, W - 0.8)
        cx, cy = (u, v) if horizontal else (v, u)
        c.rect(cx, cy, 1.2, 1.0, rng.uniform(-0.3, 0.3))
    return c.mask


def home(seed=0, size=400, width_m=10.0):
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    # tabique interior con puerta
    xw = W * rng.uniform(0.45, 0.6)
    door = rng.uniform(2.0, W - 3.0)
    c.wall(xw, 0, xw, door, 0.14)
    c.wall(xw, door + 1.0, xw, W, 0.14)
    # sala: sofá en L (cóncavo), mesa de centro, sillón, mueble TV
    sx, sy = rng.uniform(0.6, 1.0), rng.uniform(W * 0.5, W * 0.62)
    c.rect(sx + 1.1, sy, 2.2, 0.85)
    c.rect(sx + 0.42, sy + 1.1, 0.85, 1.6)
    c.rect(sx + 1.6, sy + 1.6, 1.1, 0.6)
    c.circle(sx + 2.7, sy + 2.5, 0.42)
    c.rect(W * 0.2, 0.3, 1.8, 0.4)
    # comedor: mesa redonda con sillas
    tx, ty = rng.uniform(1.4, xw - 1.4), rng.uniform(1.6, 2.6)
    c.circle(tx, ty, 0.6)
    for k in range(rng.integers(3, 6)):
        a = 2 * math.pi * k / 5 + rng.uniform(-0.2, 0.2)
        c.circle(tx + 0.95 * math.cos(a), ty + 0.95 * math.sin(a), 0.22)
    # dormitorio / cocina al otro lado
    if rng.random() < 0.5:
        c.rect(xw + 1.5, W - 1.3, 1.6, 2.0)                 # cama
        c.rect(xw + 0.4, W - 2.7, 0.45, 0.45)
        c.rect(W - 0.35, W * 0.4, 0.6, 1.8)                 # armario
    else:
        c.rect(W - 0.35, W * 0.5, 0.6, W * 0.7)             # mesón
        c.rect(xw + (W - xw) / 2, W * 0.5, 1.8, 0.9)        # isla
    c.rect(xw + (W - xw) * 0.5, 0.35, 1.2, 0.5)
    for _ in range(rng.integers(1, 4)):
        c.circle(rng.uniform(0.5, W - 0.5), rng.uniform(0.5, W - 0.5), 0.2)
    return c.mask


def workshop(seed=0, size=400, width_m=12.0):
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    for side in range(4):                                  # bancos contra las paredes
        t = 0.4
        while t < W - 1.5:
            if rng.random() < 0.5:
                L = rng.uniform(1.2, 2.5)
                if side == 0:
                    c.rect(t + L / 2, 0.35, L, 0.6)
                elif side == 1:
                    c.rect(W - 0.35, t + L / 2, 0.6, L)
                elif side == 2:
                    c.rect(t + L / 2, W - 0.35, L, 0.6)
                else:
                    c.rect(0.35, t + L / 2, 0.6, L)
                t += L
            t += rng.uniform(0.8, 1.8)
    occupied = []
    for _ in range(40):                                    # máquinas (rectángulos rotados / polígonos)
        if len(occupied) >= rng.integers(5, 9):
            break
        w, h = rng.uniform(0.7, 1.8), rng.uniform(0.6, 1.4)
        cx, cy = rng.uniform(1.8, W - 1.8), rng.uniform(1.8, W - 1.8)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if not _free_rect(occupied, box, 1.1):
            continue
        occupied.append(box)
        if rng.random() < 0.6:
            c.rect(cx, cy, w, h, rng.uniform(-0.5, 0.5))
        else:
            n = rng.integers(5, 8)
            a = np.sort(rng.uniform(0, 2 * math.pi, n))
            rr = rng.uniform(0.35, 0.7, n)
            c.poly(np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1))
    for i in (1, 2):
        c.rect(i * W / 3, W / 2 + rng.uniform(-1, 1), 0.35, 0.35)       # columnas
    return c.mask


def store(seed=0, size=400, width_m=14.0):
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    n = rng.integers(3, 5)
    aisle = (W - 3.0 - n * 1.0) / (n - 1)
    Lg = rng.uniform(0.45, 0.6) * W
    y0 = rng.uniform(1.3, 2.0)
    for i in range(n):
        x = 1.5 + i * (1.0 + aisle) + 0.5
        parts = rng.integers(1, 3)
        seg = Lg / parts
        for k in range(parts):
            c.rect(x, y0 + seg * k + seg / 2, 1.0, seg - (0.9 if parts > 1 else 0))
    # cajas registradoras y exhibidores
    for k in range(rng.integers(2, 4)):
        c.rect(1.8 + k * 2.6, W - 1.6, 1.6, 0.6)
    for _ in range(rng.integers(3, 6)):
        c.circle(rng.uniform(1.0, W - 1.0), rng.uniform(y0 + Lg + 0.9, W - 2.6), rng.uniform(0.3, 0.5))
    c.rect(W - 0.4, W / 2, 0.6, W * 0.6)                    # neveras
    return c.mask


def plaza(seed=0, size=400, width_m=18.0):
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    cx, cy = W * rng.uniform(0.4, 0.6), W * rng.uniform(0.4, 0.6)
    c.circle(cx, cy, 1.3)                                  # fuente
    a0 = rng.uniform(0, 2 * math.pi)
    c.ring(cx, cy, 3.2, 2.6, a0, a0 + rng.uniform(2.2, 3.4))   # banca curva (cóncava)
    pts = []
    for _ in range(400):                                   # árboles
        if len(pts) >= rng.integers(14, 22):
            break
        p = rng.uniform(0.8, W - 0.8, 2)
        if math.hypot(p[0] - cx, p[1] - cy) < 4.0 or any(math.hypot(*(p - q)) < 1.9 for q in pts):
            continue
        pts.append(p)
        c.circle(p[0], p[1], rng.uniform(0.25, 0.55))
    for _ in range(rng.integers(4, 8)):                    # bancas y jardineras
        x, y = rng.uniform(1, W - 1, 2)
        if math.hypot(x - cx, y - cy) < 3.8:
            continue
        if rng.random() < 0.6:
            c.rect(x, y, 1.6, 0.5, rng.uniform(0, math.pi))
        else:
            n = rng.integers(5, 8)
            a = np.sort(rng.uniform(0, 2 * math.pi, n))
            r = rng.uniform(0.5, 1.0, n)
            c.poly(np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1))
    return c.mask


def hospital(seed=0, size=400, width_m=14.0):
    """Pasillo central con habitaciones a ambos lados (puertas de 1.1 m) y camas."""
    rng = np.random.default_rng(seed)
    c = _Canvas(size, width_m)
    W = width_m
    corridor = rng.uniform(2.0, 2.6)
    y1 = W / 2 - corridor / 2
    y2 = W / 2 + corridor / 2
    n = rng.integers(3, 5)
    wroom = W / n
    for side, (ya, yb) in enumerate(((0.0, y1), (y2, W))):
        ywall = y1 if side == 0 else y2
        for k in range(n):
            x0 = k * wroom
            d = x0 + rng.uniform(0.3, wroom - 1.4)
            c.wall(x0, ywall, d, ywall)
            c.wall(d + 1.1, ywall, x0 + wroom, ywall)
            if k > 0:
                c.wall(x0, ya, x0, yb)
            # cama y mesa de noche
            bx = x0 + wroom / 2 + rng.uniform(-0.3, 0.3)
            by = (ya + 1.2) if side == 0 else (yb - 1.2)
            c.rect(bx, by, 1.0, 2.0)
            c.rect(bx + 0.9, by, 0.45, 0.45)
    for _ in range(rng.integers(1, 3)):                    # camillas en el pasillo
        c.rect(rng.uniform(1.5, W - 1.5), y1 + 0.45, 1.9, 0.6)
    return c.mask


SCENES = {
    "office": (office, 12.0),
    "warehouse": (warehouse, 16.0),
    "home": (home, 10.0),
    "workshop": (workshop, 12.0),
    "store": (store, 14.0),
    "plaza": (plaza, 18.0),
    "hospital": (hospital, 14.0),
}

LABELS = {"office": "Office", "warehouse": "Warehouse", "home": "Apartment", "workshop": "Workshop",
          "store": "Retail store", "plaza": "Outdoor plaza", "hospital": "Hospital ward"}
