"""Generadores de entornos para simulación y pruebas.

Todos devuelven un ``Environment`` (obstáculos en blanco = True).  El tamaño
se da en píxeles ``(alto, ancho)`` o como entero (cuadrado) y la escala se
fija con ``width_m`` (ancho físico).
"""
from __future__ import annotations

import cv2
import numpy as np

from common.environment import Environment


def _canvas(size):
    if isinstance(size, int):
        size = (size, size)
    return np.zeros(size, dtype=np.uint8)


def _env(img, width_m, name):
    return Environment(img > 0, width_m / img.shape[1], name=name)


def _regular_polygon(cx, cy, r, n, rot):
    ang = rot + np.arange(n) * 2 * np.pi / n
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1).round().astype(np.int32)


def _random_convex(rng, cx, cy, r):
    n = rng.integers(3, 9)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = r * rng.uniform(0.6, 1.0, n)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1)
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    return hull.round().astype(np.int32)


def paper_environment(size: int = 480, width_m: float = 20.0) -> Environment:
    """Aproximación del entorno de la figura 2a (2 hexágonos, 2 rectángulos, 1 círculo)."""
    img = _canvas(size)
    s = size / 480.0
    cv2.fillPoly(img, [_regular_polygon(118 * s, 97 * s, 66 * s, 6, np.pi / 2)], 255)
    cv2.fillPoly(img, [_regular_polygon(330 * s, 347 * s, 92 * s, 6, np.pi / 2)], 255)
    cv2.rectangle(img, (int(305 * s), int(85 * s)), (int(408 * s), int(178 * s)), 255, -1)
    cv2.rectangle(img, (int(164 * s), int(206 * s)), (int(236 * s), int(263 * s)), 255, -1)
    cv2.ellipse(img, (int(88 * s), int(331 * s)), (int(45 * s), int(41 * s)), 0, 0, 360, 255, -1)
    return _env(img, width_m, "articulo")


def paper_like(seed: int = 0, size: int = 480, width_m: float = 20.0, n: int | None = None) -> Environment:
    """Hexágonos, rectángulos y círculos separados (estilo del artículo)."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    n = n or int(rng.integers(4, 9))
    placed = []
    tries = 0
    while len(placed) < n and tries < 500:
        tries += 1
        r = rng.uniform(0.06, 0.16) * min(h, w)
        cx = rng.uniform(r + 12, w - r - 12)
        cy = rng.uniform(r + 12, h - r - 12)
        if any(np.hypot(cx - px, cy - py) < r + pr + 0.06 * min(h, w) for px, py, pr in placed):
            continue
        placed.append((cx, cy, r))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.fillPoly(img, [_regular_polygon(cx, cy, r, 6, rng.uniform(0, np.pi))], 255)
        elif kind == 1:
            a = r * rng.uniform(0.6, 1.0)
            b = r * rng.uniform(0.5, 1.0)
            cv2.rectangle(img, (int(cx - a), int(cy - b)), (int(cx + a), int(cy + b)), 255, -1)
        else:
            cv2.circle(img, (int(cx), int(cy)), int(r * 0.85), 255, -1)
    return _env(img, width_m, f"articulo_aleatorio_{seed}")


def random_shapes(seed: int = 0, size=400, width_m: float = 20.0, n: int = 12,
                  allow_overlap: bool = True) -> Environment:
    """Polígonos convexos, elipses y rectángulos rotados; pueden superponerse
    y formar obstáculos no convexos."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    for _ in range(n):
        r = rng.uniform(0.03, 0.12) * min(h, w)
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        k = rng.integers(0, 3)
        if k == 0:
            cv2.fillPoly(img, [_random_convex(rng, cx, cy, r)], 255)
        elif k == 1:
            cv2.ellipse(img, (int(cx), int(cy)), (int(r), int(r * rng.uniform(0.3, 1))),
                        float(rng.uniform(0, 180)), 0, 360, 255, -1)
        else:
            box = cv2.boxPoints(((cx, cy), (r * 2, r * rng.uniform(0.2, 1.0)), rng.uniform(0, 180)))
            cv2.fillPoly(img, [box.round().astype(np.int32)], 255)
    return _env(img, width_m, f"formas_{seed}")


def cluttered(seed: int = 0, size=400, width_m: float = 20.0, n: int = 40) -> Environment:
    """Muchos obstáculos pequeños (bosque)."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    for _ in range(n):
        r = rng.uniform(0.012, 0.035) * min(h, w)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        if rng.random() < 0.5:
            cv2.circle(img, (int(cx), int(cy)), max(2, int(r)), 255, -1)
        else:
            cv2.fillPoly(img, [_random_convex(rng, cx, cy, r * 1.3)], 255)
    return _env(img, width_m, f"bosque_{seed}")


def concave(seed: int = 0, size=400, width_m: float = 20.0) -> Environment:
    """Obstáculos en U, L y espiral (concavidades / trampas locales)."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    t = max(4, int(0.025 * min(h, w)))
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        a = rng.uniform(0.08, 0.18) * min(h, w)
        kind = rng.integers(0, 3)
        rot = rng.integers(0, 4)
        if kind == 0:     # U
            pts = [[-a, -a], [-a, a], [a, a], [a, -a]]
        elif kind == 1:   # L
            pts = [[-a, -a], [-a, a], [a, a]]
        else:             # espiral cuadrada
            pts = [[-a, -a], [a, -a], [a, a], [-a * 0.6, a], [-a * 0.6, -a * 0.4], [a * 0.4, -a * 0.4],
                   [a * 0.4, a * 0.5]]
        P = np.asarray(pts, dtype=float)
        for _ in range(rot):
            P = P @ np.array([[0, -1], [1, 0]])
        P = (P + [cx, cy]).round().astype(np.int32)
        cv2.polylines(img, [P], False, 255, t)
    return _env(img, width_m, f"concavos_{seed}")


def rooms(seed: int = 0, size=400, width_m: float = 20.0, nx: int = 3, ny: int = 3,
          door: float = 0.09) -> Environment:
    """Plano de interiores: habitaciones con puertas; paredes unidas al marco."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    t = max(3, int(0.015 * min(h, w)))
    dw = int(door * min(h, w))
    xs = np.linspace(0, w, nx + 1).astype(int)
    ys = np.linspace(0, h, ny + 1).astype(int)
    for x in xs[1:-1]:
        for j in range(ny):
            y0, y1 = ys[j], ys[j + 1]
            cv2.line(img, (x, y0), (x, y1), 255, t)
            if rng.random() < 0.85:
                d = int(rng.uniform(y0 + dw, y1 - 2 * dw))
                cv2.line(img, (x, d), (x, d + dw), 0, t + 2)
    for y in ys[1:-1]:
        for i in range(nx):
            x0, x1 = xs[i], xs[i + 1]
            cv2.line(img, (x0, y), (x1, y), 255, t)
            if rng.random() < 0.85:
                d = int(rng.uniform(x0 + dw, x1 - 2 * dw))
                cv2.line(img, (d, y), (d + dw, y), 0, t + 2)
    # mobiliario
    for _ in range(int(rng.integers(3, 8))):
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        a, b = rng.uniform(0.02, 0.05, 2) * min(h, w)
        cv2.rectangle(img, (int(cx - a), int(cy - b)), (int(cx + a), int(cy + b)), 255, -1)
    return _env(img, width_m, f"habitaciones_{seed}")


def maze(seed: int = 0, size=400, width_m: float = 20.0, cells: int = 6) -> Environment:
    """Laberinto perfecto (todas las paredes conectadas al marco)."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    t = max(3, int(0.02 * min(h, w)))
    cw, ch = w / cells, h / cells
    visited = np.zeros((cells, cells), bool)
    walls_v = np.ones((cells, cells + 1), bool)
    walls_h = np.ones((cells + 1, cells), bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= r + dr < cells and 0 <= c + dc < cells and not visited[r + dr, c + dc]]
        if not nb:
            stack.pop()
            continue
        nr, nc = nb[int(rng.integers(len(nb)))]
        if nr != r:
            walls_h[max(r, nr), c] = False
        else:
            walls_v[r, max(c, nc)] = False
        visited[nr, nc] = True
        stack.append((nr, nc))
    for r in range(cells):
        for c in range(cells + 1):
            if walls_v[r, c]:
                cv2.line(img, (int(c * cw), int(r * ch)), (int(c * cw), int((r + 1) * ch)), 255, t)
    for r in range(cells + 1):
        for c in range(cells):
            if walls_h[r, c]:
                cv2.line(img, (int(c * cw), int(r * ch)), (int((c + 1) * cw), int(r * ch)), 255, t)
    return _env(img, width_m, f"laberinto_{seed}")


def narrow_passages(seed: int = 0, size=400, width_m: float = 20.0, gap_m: float = 0.7) -> Environment:
    """Muros transversales con aberturas estrechas (apenas mayores que el robot)."""
    rng = np.random.default_rng(seed)
    img = _canvas(size)
    h, w = img.shape
    mpp = width_m / w
    gap = max(2, int(gap_m / mpp))
    t = max(4, int(0.03 * min(h, w)))
    for k in range(1, 4):
        x = int(k * w / 4)
        g0 = int(rng.uniform(0.1, 0.9 - gap / h) * h)
        cv2.line(img, (x, int(0.08 * h)), (x, g0), 255, t)
        cv2.line(img, (x, g0 + gap), (x, int(0.92 * h)), 255, t)
    return _env(img, width_m, f"pasos_estrechos_{seed}")


def empty(size=300, width_m: float = 15.0) -> Environment:
    return _env(_canvas(size), width_m, "vacio")


def single_obstacle(size=300, width_m: float = 15.0) -> Environment:
    img = _canvas(size)
    h, w = img.shape
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 6, 255, -1)
    return _env(img, width_m, "un_obstaculo")


def disconnected(size=300, width_m: float = 15.0) -> Environment:
    """Muro completo que divide el entorno en dos regiones sin conexión."""
    img = _canvas(size)
    h, w = img.shape
    cv2.line(img, (w // 2, 0), (w // 2, h), 255, max(4, w // 40))
    return _env(img, width_m, "desconectado")


GENERATORS = {
    "articulo_aleatorio": paper_like,
    "formas": random_shapes,
    "bosque": cluttered,
    "concavos": concave,
    "habitaciones": rooms,
    "laberinto": maze,
    "pasos_estrechos": narrow_passages,
}


def random_free_points(occ: np.ndarray, rng: np.random.Generator, n: int = 2,
                       min_sep: float = 0.0, max_tries: int = 1000) -> np.ndarray | None:
    ys, xs = np.nonzero(~occ)
    if len(ys) == 0:
        return None
    for _ in range(max_tries):
        idx = rng.integers(0, len(ys), n)
        pts = np.stack([xs[idx], ys[idx]], axis=1).astype(float)
        if n < 2 or all(np.hypot(*(pts[i] - pts[j])) >= min_sep for i in range(n) for j in range(i + 1, n)):
            return pts
    return None
