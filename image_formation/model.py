"""Fotos sintéticas realistas con verdad de terreno conocida.

Se usan para *garantizar* el procesamiento de percepción: como se conoce
exactamente qué píxeles son obstáculo, se puede medir la calidad de la
segmentación y, sobre todo, verificar que la ruta planificada con el mapa
detectado no choca con los obstáculos reales.

Efectos simulados: texturas de piso (uniforme, alfombra, madera, baldosa,
concreto), objetos con color/sombreado/bisel/textura, sombras proyectadas,
iluminación no uniforme (gradiente + viñeteo), ruido de sensor, desenfoque,
compresión JPEG y perspectiva de cámara inclinada.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

TEXTURES = ("uniforme", "alfombra", "madera", "baldosa", "concreto")


@dataclass
class PhotoConditions:
    texture: str = "madera"
    min_contrast: float = 25.0        # ΔE (Lab) mínimo entre objetos y piso
    shadow: float = 0.35              # oscurecimiento máximo de la sombra (0 = sin sombras)
    shadow_offset: float = 0.02       # desplazamiento de la sombra (fracción del lado)
    light_gradient: float = 0.2       # variación de iluminación (±)
    vignette: float = 0.15
    noise: float = 0.02               # desviación del ruido gaussiano (0..1)
    blur: float = 0.6                 # sigma de desenfoque
    jpeg: int = 80                    # calidad JPEG (0 = sin comprimir)
    perspective: float = 0.0          # 0 = cenital; 0.1-0.25 = cámara inclinada
    object_texture: float = 0.5       # probabilidad de que un objeto tenga textura/estampado


@dataclass
class SyntheticPhoto:
    photo: np.ndarray                 # RGB uint8 (lo que "toma la cámara")
    gt: np.ndarray                    # bool, obstáculos en el marco rectificado
    corners: np.ndarray | None        # 4 esquinas del piso en la foto (si hay perspectiva)
    scene: np.ndarray                 # RGB uint8 cenital sin perspectiva
    conditions: PhotoConditions = field(default_factory=PhotoConditions)


LEVELS = {
    "facil": dict(textures=("uniforme", "alfombra"), min_contrast=(40, 60), shadow=(0.0, 0.15),
                  light_gradient=(0.0, 0.1), vignette=(0.0, 0.08), noise=(0.005, 0.015), blur=(0.0, 0.5),
                  jpeg=(85, 95), perspective=(0.0, 0.0)),
    "medio": dict(textures=TEXTURES, min_contrast=(25, 40), shadow=(0.2, 0.4),
                  light_gradient=(0.1, 0.25), vignette=(0.05, 0.2), noise=(0.01, 0.025), blur=(0.3, 0.9),
                  jpeg=(70, 90), perspective=(0.0, 0.12)),
    "dificil": dict(textures=("madera", "baldosa", "concreto", "alfombra"), min_contrast=(15, 25),
                    shadow=(0.35, 0.55), light_gradient=(0.25, 0.4), vignette=(0.15, 0.3),
                    noise=(0.025, 0.04), blur=(0.6, 1.3), jpeg=(50, 70), perspective=(0.1, 0.22)),
}


def random_conditions(level: str, rng: np.random.Generator, levels: dict | None = None) -> PhotoConditions:
    L = (levels or LEVELS)[level]      # levels: from configs/image_formation.json
    u = lambda key: float(rng.uniform(*L[key]))  # noqa: E731
    return PhotoConditions(texture=str(rng.choice(L["textures"])), min_contrast=u("min_contrast"),
                           shadow=u("shadow"), light_gradient=u("light_gradient"), vignette=u("vignette"),
                           noise=u("noise"), blur=u("blur"), jpeg=int(u("jpeg")), perspective=u("perspective"))


# --------------------------------------------------------------------------- #
def _value_noise(h, w, cell, rng, octaves=1):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cell / (2 ** o)))
        small = rng.random((h // c + 3, w // c + 3)).astype(np.float32)
        big = cv2.resize(small, (w + 3 * c, h + 3 * c), interpolation=cv2.INTER_CUBIC)[:h, :w]
        out += amp * big
        total += amp
        amp *= 0.5
    return out / total


def _lab_to_rgb(L, a, b):
    lab = np.array([[[L * 255 / 100, a + 128, b + 128]]], np.float32).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0].astype(np.float32) / 255.0


def _rgb_to_lab(rgb01):
    px = (np.asarray(rgb01, np.float32).reshape(1, 1, 3) * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128])


def floor_image(h, w, kind, rng) -> np.ndarray:
    """Textura de piso RGB float 0..1."""
    base_L = rng.uniform(35, 85)
    base = _lab_to_rgb(base_L, rng.uniform(-8, 15), rng.uniform(-5, 30))
    img = np.ones((h, w, 3), np.float32) * base
    if kind == "uniforme":
        img *= (1 + 0.02 * (_value_noise(h, w, 40, rng) - 0.5))[..., None]
    elif kind == "alfombra":
        fine = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8)
        img *= (1 + 0.06 * fine + 0.05 * (_value_noise(h, w, 60, rng) - 0.5))[..., None]
    elif kind == "madera":
        horizontal = rng.random() < 0.5
        H, W = (h, w) if horizontal else (w, h)
        plank = int(rng.uniform(0.07, 0.14) * min(h, w))
        tex = np.ones((H, W), np.float32)
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        warp = _value_noise(H, W, 30, rng) * 6
        grain = np.sin((y + warp) * rng.uniform(0.5, 0.9) + 0.02 * x) * 0.5 + 0.5
        tex *= 1 + 0.07 * (grain - 0.5)
        for top in range(0, H, plank):
            tex[top:top + plank] *= 1 + rng.uniform(-0.08, 0.08)
            tex[top:top + 1] *= 0.75                        # junta entre tablas
            off = int(rng.uniform(0, W))
            tex[top:top + plank, off:off + 1] *= 0.8
        if not horizontal:
            tex = tex.T
        img *= tex[..., None]
    elif kind == "baldosa":
        tile = int(rng.uniform(0.08, 0.16) * min(h, w))
        grout = max(1, int(tile * 0.05))
        tex = np.ones((h, w), np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((yy // tile + xx // tile) % 2).astype(np.float32)
        tex *= 1 + rng.uniform(0.0, 0.08) * (checker - 0.5)
        tex *= 1 + 0.03 * (_value_noise(h, w, 25, rng) - 0.5)
        g = ((yy % tile) < grout) | ((xx % tile) < grout)
        gv = rng.choice([0.7, 1.15])
        tex[g] *= gv
        img *= tex[..., None]
    elif kind == "concreto":
        n = _value_noise(h, w, 80, rng, octaves=4)
        speck = (rng.random((h, w)) < 0.01).astype(np.float32)
        speck = cv2.GaussianBlur(speck, (0, 0), 0.7) * 3
        img *= (1 + 0.14 * (n - 0.5) - 0.12 * speck)[..., None]
    else:
        raise ValueError(kind)
    return img.clip(0, 1)


def _object_color(floor_lab, min_contrast, rng):
    for _ in range(300):
        L = rng.uniform(10, 95)
        a = rng.uniform(-50, 60)
        b = rng.uniform(-50, 70)
        rgb = _lab_to_rgb(L, a, b)
        lab = _rgb_to_lab(rgb)
        if np.linalg.norm(lab - floor_lab) >= min_contrast:
            return rgb
    return _lab_to_rgb(5 if floor_lab[0] > 50 else 95, 0, 0)


def render_photo(gt: np.ndarray, cond: PhotoConditions, rng: np.random.Generator,
                 photo_size: tuple[int, int] = (480, 640)) -> SyntheticPhoto:
    """Genera la foto de una escena cuyo mapa de obstáculos real es ``gt``."""
    gt = gt.astype(bool)
    h, w = gt.shape
    floor = floor_image(h, w, cond.texture, rng)
    floor_lab = _rgb_to_lab(floor.reshape(-1, 3).mean(0))
    img = floor.copy()

    # sombras proyectadas (sobre el piso)
    if cond.shadow > 0:
        ang = rng.uniform(0, 2 * np.pi)
        off = cond.shadow_offset * min(h, w) * rng.uniform(0.6, 1.4)
        M = np.float32([[1, 0, off * np.cos(ang)], [0, 1, off * np.sin(ang)]])
        sh = cv2.warpAffine(gt.astype(np.float32), M, (w, h))
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, off * 0.35))
        img *= (1 - cond.shadow * sh)[..., None]

    # objetos
    n, lab = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        m = lab == i
        col = _object_color(floor_lab, cond.min_contrast, rng)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = m[y0:y1, x0:x1]
        hh, ww = sub.shape
        obj = np.ones((hh, ww, 3), np.float32) * col
        # sombreado direccional + bisel (bordes más oscuros)
        gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        shade = 1 + rng.uniform(0.05, 0.15) * ((gx / max(ww, 1) - 0.5) * np.cos(ang) + (gy / max(hh, 1) - 0.5) * np.sin(ang))
        dist = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
        bevel = 1 - 0.25 * np.exp(-dist / 2.5)
        obj *= (shade * bevel)[..., None]
        if rng.random() < cond.object_texture:
            kind = rng.integers(0, 3)
            if kind == 0:     # rayas
                period = rng.uniform(6, 16)
                stripes = (np.sin((gx * np.cos(ang) + gy * np.sin(ang)) * 2 * np.pi / period) > 0.6)
                obj[stripes] *= rng.uniform(0.7, 0.9)
            elif kind == 1:   # ruido (tela, material)
                obj *= (1 + 0.12 * (_value_noise(hh, ww, 6, rng) - 0.5))[..., None]
            else:             # cinta / etiqueta
                t = max(2, int(0.12 * min(hh, ww)))
                c = rng.integers(0, max(1, min(hh, ww) - t))
                if rng.random() < 0.5:
                    obj[c:c + t, :] = obj[c:c + t, :] * 0.2 + 0.8 * _object_color(floor_lab, 10, rng)
                else:
                    obj[:, c:c + t] = obj[:, c:c + t] * 0.2 + 0.8 * _object_color(floor_lab, 10, rng)
        region = img[y0:y1, x0:x1]
        region[sub] = obj[sub]

    # iluminación no uniforme
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx / w - 0.5
    v = yy / h - 0.5
    ang = rng.uniform(0, 2 * np.pi)
    light = 1 + cond.light_gradient * 2 * (u * np.cos(ang) + v * np.sin(ang)) - cond.vignette * 2 * (u * u + v * v)
    img *= light[..., None]
    scene = (img.clip(0, 1) * 255).astype(np.uint8)

    corners = None
    photo = img
    if cond.perspective > 0:
        ph, pw = photo_size
        # cuadrilátero del piso en la foto: trapecio (cámara inclinada) + giro leve
        s = 0.8 * min(pw / w, ph / h)
        cw, ch = w * s, h * s
        p = cond.perspective
        top_shrink = p * cw * rng.uniform(0.7, 1.3)
        quad = np.float32([[-cw / 2 + top_shrink, -ch / 2], [cw / 2 - top_shrink, -ch / 2],
                           [cw / 2, ch / 2], [-cw / 2, ch / 2]])
        quad[:, 1] *= 1 - 0.5 * p                         # escorzo vertical
        rot = rng.uniform(-0.15, 0.15)
        R = np.float32([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        quad = quad @ R.T + np.float32([pw / 2, ph / 2]) + rng.uniform(-0.03, 0.03, 2) * [pw, ph]
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        Hm = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        wall = floor_image(ph, pw, "concreto", rng) * rng.uniform(0.4, 0.9)
        warped = cv2.warpPerspective(img, Hm, (pw, ph), flags=cv2.INTER_LINEAR)
        inside = cv2.warpPerspective(np.ones((h, w), np.float32), Hm, (pw, ph)) > 0.5
        photo = np.where(inside[..., None], warped, wall)
        corners = quad

    # sensor: desenfoque, ruido, JPEG
    if cond.blur > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), cond.blur)
    if cond.noise > 0:
        photo = photo + rng.normal(0, cond.noise, photo.shape).astype(np.float32)
    photo = (np.clip(photo, 0, 1) * 255).astype(np.uint8)
    if cond.jpeg:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, cond.jpeg])
        photo = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return SyntheticPhoto(photo, gt.copy(), corners, scene, cond)
