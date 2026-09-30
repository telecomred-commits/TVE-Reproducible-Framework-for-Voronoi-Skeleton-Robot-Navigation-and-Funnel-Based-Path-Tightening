"""Percepción: de una foto de un entorno real a un mapa de obstáculos (M_b).

Pensado para fotos **cenitales** (tomadas desde arriba) o fotos inclinadas de un
piso plano que se rectifican con 4 esquinas de un rectángulo de medidas
conocidas.  Métodos disponibles (todos clásicos, sin redes neuronales):

``piso`` (recomendado)
    Modelo estadístico del color del piso en el espacio Lab.  El piso se estima
    automáticamente (grupo de color dominante, unido con los grupos vecinos en
    color, p. ej. gradientes de luz o baldosas de dos tonos) o a partir de
    muestras marcadas por el usuario.  Cada píxel se clasifica por su distancia
    de Mahalanobis al modelo del piso.  El canal de luminosidad L tiene un peso
    menor (``shadow_weight``) para no confundir sombras con obstáculos, y la
    iluminación desigual se corrige ajustando una superficie cuadrática al piso.
``kmeans``
    Agrupamiento de colores; todo lo que no pertenece a los grupos del piso es
    obstáculo (asignación dura).
``bordes``
    Bordes de Canny cerrados morfológicamente y rellenados.  Útil con objetos
    de color parecido al piso pero con contornos definidos.
``umbral``
    Otsu sobre la luminosidad; la clase minoritaria es obstáculo.
``combinado``
    Unión de ``piso`` y ``bordes``.

Todas terminan con el mismo post-proceso morfológico: apertura (quita ruido y
juntas de baldosas), cierre, relleno de huecos pequeños, eliminación de
manchas pequeñas y, opcionalmente, envolvente convexa por objeto.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from common.environment import Environment

METHODS = ("piso", "kmeans", "bordes", "umbral", "combinado")


@dataclass
class PerceptionParams:
    method: str = "piso"
    max_side: int = 500              # lado mayor de la imagen procesada [px] (= resolución del mapa)
    denoise: int = 7                 # diámetro del filtro bilateral (0 = no filtrar)
    shadow_weight: float = 0.5       # peso de L frente a (a, b); menor = más tolerante a sombras
    asymmetric_light: bool = True    # el aclarado usa peso completo de L (las sombras solo oscurecen)
    threshold: float = 3.0           # umbral de distancia de Mahalanobis al piso
    color_noise: float = 2.0         # ruido de color mínimo del piso [unidades Lab] (regulariza la covarianza)
    illumination: bool = True        # corrección de iluminación (superficie cuadrática del piso)
    local_contrast: bool = True      # distancia a la mediana local (objetos pequeños / poco contraste)
    local_window: float = 0.08       # ventana de la mediana local (fracción del lado mayor)
    local_factor: float = 1.0        # umbral de la señal local = local_factor × threshold
    hysteresis: float = 1.0          # umbral bajo = hysteresis × umbral (1 = sin histéresis)
    k_clusters: int = 6
    floor_merge: float = 3.0         # grupos a esta distancia del piso también son piso
    open_px: int = 1                 # radio de apertura morfológica
    close_px: int = 3                # radio de cierre morfológico
    fill_holes: bool = True
    max_hole_frac: float = 0.01      # solo se rellenan huecos menores a esta fracción de la imagen
    min_area_frac: float = 0.0001    # se eliminan manchas menores a esta fracción
    convex_hull: bool = False
    extra_dilate_px: int = 2         # margen de percepción (absorbe el error de borde; ver simulaciones)
    floor_samples: list = field(default_factory=list)   # [(x, y)] en la imagen procesada
    sample_radius: int = 7
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerceptionResult:
    image: np.ndarray                # RGB uint8 procesada (rectificada / redimensionada)
    mask: np.ndarray                 # bool, True = obstáculo
    distance: np.ndarray | None      # float32, distancia al modelo del piso (None si no aplica)
    raw_mask: np.ndarray             # máscara antes del post-proceso
    floor_seed: np.ndarray | None    # bool, píxeles usados para el modelo del piso
    n_objects: int
    timings: dict
    params: PerceptionParams

    def overlay(self, alpha: float = 0.5, color=(255, 40, 40)) -> np.ndarray:
        out = self.image.astype(np.float32).copy()
        c = np.asarray(color, np.float32)
        out[self.mask] = (1 - alpha) * out[self.mask] + alpha * c
        cnts, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = out.clip(0, 255).astype(np.uint8)
        cv2.drawContours(out, cnts, -1, (255, 230, 0), 1)
        return out


# --------------------------------------------------------------------------- #
# Carga, rectificación y redimensionado
# --------------------------------------------------------------------------- #
def load_photo(path: str | Path) -> np.ndarray:
    """Lee una foto (rutas con tildes incluidas) y la devuelve en RGB uint8."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def order_corners(pts) -> np.ndarray:
    """Ordena 4 puntos como sup-izq, sup-der, inf-der, inf-izq."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]                    # sentido horario en coordenadas de imagen
    start = int(np.argmin(p.sum(axis=1)))     # sup-izq = menor x + y
    return np.roll(p, -start, axis=0)


def rectify(img: np.ndarray, corners, width_m: float, height_m: float, max_side: int = 500):
    """Corrige la perspectiva: el cuadrilátero ``corners`` pasa a ser un
    rectángulo con la proporción real ``width_m`` × ``height_m``.

    Devuelve (imagen_rectificada, metros_por_píxel, homografía).
    """
    src = order_corners(corners)
    if width_m >= height_m:
        w = int(max_side)
        h = max(8, int(round(max_side * height_m / width_m)))
    else:
        h = int(max_side)
        w = max(8, int(round(max_side * width_m / height_m)))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return out, width_m / w, H


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if abs(s - 1) < 1e-6:
        return img.copy()
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=interp)


# --------------------------------------------------------------------------- #
# Características y modelo del piso
# --------------------------------------------------------------------------- #
def lab_features(rgb: np.ndarray, shadow_weight: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV (uint8): L en 0..255, a y b desplazados en 128 -> escala CIE aproximada
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return np.stack([L * shadow_weight, a, b], axis=-1)


def _kmeans(X: np.ndarray, k: int, seed: int):
    k = int(max(1, min(k, len(X))))
    cv2.setRNGSeed(int(seed))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, lab, centers = cv2.kmeans(X.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return lab.ravel(), centers


@dataclass
class _Gauss:
    mu: np.ndarray
    icov: np.ndarray


def _fit_gauss(X: np.ndarray, reg: float = 4.0) -> _Gauss:
    mu = X.mean(axis=0)
    cov = np.cov(X.T) if len(X) > 3 else np.eye(X.shape[1])
    cov = np.atleast_2d(cov) + reg * np.eye(X.shape[1])
    return _Gauss(mu, np.linalg.inv(cov))


def _mahal(F: np.ndarray, g: _Gauss, bright_gain: float = 1.0) -> np.ndarray:
    """Distancia de Mahalanobis.  ``bright_gain`` amplifica solo el aumento de
    luminosidad: las sombras oscurecen pero nunca aclaran, así que un píxel más
    claro que el piso se evalúa con el peso completo de L."""
    D = F - g.mu
    if bright_gain != 1.0:
        D = D.copy()
        D[..., 0] = np.where(D[..., 0] > 0, D[..., 0] * bright_gain, D[..., 0])
    return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", D, g.icov, D), 0))


def floor_models(F: np.ndarray, params: PerceptionParams, rng: np.random.Generator):
    """Modelos gaussianos del piso y máscara de los píxeles semilla usados."""
    h, w, _ = F.shape
    flat = F.reshape(-1, 3)
    seed_mask = np.zeros((h, w), bool)
    if params.floor_samples:
        r = max(1, int(params.sample_radius))
        parts = []
        for x, y in params.floor_samples:
            disk = np.zeros((h, w), np.uint8)
            cv2.circle(disk, (int(round(x)), int(round(y))), r, 1, -1)
            Xi = F[disk > 0]
            if len(Xi) == 0:
                continue
            # Filtro robusto: si el disco toca un objeto, se descartan los
            # píxeles lejanos a la mediana del disco (el color del clic).
            med = np.median(Xi, axis=0)
            dev = np.linalg.norm(Xi - med, axis=1)
            mad = np.median(dev) + 1e-6
            keep = dev <= max(3.0 * mad, 2.0)
            parts.append(Xi[keep])
            ys, xs = np.nonzero(disk)
            seed_mask[ys[keep], xs[keep]] = True
        X = np.vstack(parts) if parts else np.zeros((0, 3), np.float32)
        if len(X) < 5:
            raise ValueError("Muestras de piso insuficientes")
        k = min(3, len(params.floor_samples))
        labels, _ = _kmeans(X, k, params.seed)
        return [_fit_gauss(X[labels == i], params.color_noise ** 2) for i in range(k) if (labels == i).sum() >= 5], seed_mask

    # Automático: grupo dominante + grupos cercanos en color (unión transitiva)
    n = len(flat)
    idx = rng.choice(n, size=min(n, 30000), replace=False)
    X = flat[idx]
    labels, centers = _kmeans(X, params.k_clusters, params.seed)
    counts = np.bincount(labels, minlength=len(centers))
    gs = [_fit_gauss(X[labels == i], params.color_noise ** 2) if counts[i] >= 5 else None for i in range(len(centers))]
    main = int(np.argmax(counts))
    floor = {main}

    gain = _bright_gain(params)

    def dist(i, j):
        # Distancia del candidato i al grupo de piso j con la MISMA métrica que la
        # detección (un aclarado pesa más que un oscurecimiento), evaluada con la
        # covarianza de ambos grupos (se toma la menor).
        d = centers[i] - centers[j]
        if d[0] > 0:
            d = d * np.array([gain, 1.0, 1.0], np.float32)
        return float(min(np.sqrt(max(d @ gs[j].icov @ d, 0)), np.sqrt(max(d @ gs[i].icov @ d, 0))))

    changed = True
    while changed:
        changed = False
        for i in range(len(centers)):
            if i in floor or gs[i] is None:
                continue
            # La unión puede encadenarse (gradientes de luz, baldosas de dos
            # tonos), pero sin alejarse demasiado del grupo principal: si no,
            # piso -> sombra -> borde difuso -> objeto terminaría siendo "piso".
            if dist(i, main) > 2.0 * params.floor_merge:
                continue
            if any(dist(i, j) < params.floor_merge for j in floor):
                floor.add(i)
                changed = True
    models = [gs[i] for i in sorted(floor) if gs[i] is not None]
    # máscara semilla (solo para visualizar): píxeles cuyo grupo es de piso
    full_lab = np.argmin(((flat[:, None, :] - centers[None]) ** 2).sum(-1), axis=1).reshape(h, w)
    seed_mask = np.isin(full_lab, sorted(floor))
    return models, seed_mask


def _illumination_correct(F: np.ndarray, floor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resta a cada canal una superficie cuadrática ajustada sobre el piso."""
    h, w, _ = F.shape
    ys, xs = np.nonzero(floor)
    if len(ys) < 50:
        return F
    sel = rng.choice(len(ys), size=min(len(ys), 20000), replace=False)
    ys, xs = ys[sel], xs[sel]
    u = xs / max(w - 1, 1) - 0.5
    v = ys / max(h - 1, 1) - 0.5
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)
    gy, gx = np.mgrid[0:h, 0:w]
    U = gx / max(w - 1, 1) - 0.5
    V = gy / max(h - 1, 1) - 0.5
    B = np.stack([np.ones_like(U), U, V, U * U, U * V, V * V], axis=-1)
    out = F.copy()
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, F[ys, xs, c], rcond=None)
        surf = B @ coef
        out[..., c] = F[..., c] - (surf - coef[0])     # conserva el nivel medio
    return out


# --------------------------------------------------------------------------- #
# Métodos
# --------------------------------------------------------------------------- #
def _local_distance(rgb, F, floor_px, params):
    """Distancia de Mahalanobis del residuo F - mediana_local(F).

    Para objetos más pequeños que media ventana, la mediana local es el piso
    que los rodea: el residuo es el contraste real del objeto, sin gradientes
    de iluminación ni variaciones lentas del piso.
    """
    k = int(params.local_window * max(rgb.shape[:2])) | 1
    k = max(k, 7)
    med = np.stack([cv2.medianBlur(rgb[..., c], k) for c in range(3)], axis=-1)
    R = F - lab_features(med, params.shadow_weight)
    X = R[floor_px]
    if len(X) < 50:
        return np.zeros(F.shape[:2], np.float32)
    g = _fit_gauss(X, reg=2.0)
    g.mu = np.zeros(3)                       # el residuo del piso es de media cero
    return _mahal(R, g, _bright_gain(params))


def _bright_gain(params) -> float:
    return 1.0 / max(params.shadow_weight, 1e-3) if params.asymmetric_light else 1.0


def _hysteresis(D, high, low):
    strong = D > high
    weak = D > low
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def _method_floor(F, params, rng, rgb=None):
    models, seed = floor_models(F, params, rng)
    D = np.min(np.stack([_mahal(F, g) for g in models]), axis=0)
    if params.illumination:
        # 2ª pasada: corregir iluminación con los píxeles claramente de piso
        floor_px = D < params.threshold * 0.6
        F2 = _illumination_correct(F, floor_px, rng)
        models, seed = floor_models(F2, params, rng)
        F = F2
    D = np.min(np.stack([_mahal(F, g, _bright_gain(params)) for g in models]), axis=0)
    th = params.threshold
    if params.hysteresis < 1.0:
        mask = _hysteresis(D, th, params.hysteresis * th)
    else:
        mask = D > th
    if params.local_contrast and rgb is not None:
        # Solo se aceptan de la señal local componentes PEQUEÑOS y SEPARADOS de
        # lo ya detectado: así se recuperan objetos pequeños de poco contraste
        # sin crear halos alrededor de los objetos grandes.
        Dl = _local_distance(rgb, F, D < th * 0.6, params)
        k = max(7, int(params.local_window * max(rgb.shape[:2])))
        small_lim = 0.25 * k * k
        loc = (Dl > th * params.local_factor).astype(np.uint8)
        # objetos globales grandes: junto a ellos la señal local produce halos
        ng, glab, gstats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        big = np.zeros(ng, bool)
        big[1:] = gstats[1:, cv2.CC_STAT_AREA] > small_lim
        near_big = cv2.dilate(big[glab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(loc, connectivity=8)
        ok = np.zeros(n, bool)
        ok[1:] = stats[1:, cv2.CC_STAT_AREA] <= small_lim
        ok[np.unique(lab[near_big & (lab > 0)])] = False
        ok[0] = False
        mask = mask | ok[lab]
        D = np.maximum(D, np.where(ok[lab], Dl, 0))
    return mask, D.astype(np.float32), seed


def _method_kmeans(F, params, rng):
    h, w, _ = F.shape
    models, seed = floor_models(F, params, rng)
    flat = F.reshape(-1, 3)
    idx = rng.choice(len(flat), size=min(len(flat), 30000), replace=False)
    labels, centers = _kmeans(flat[idx], params.k_clusters, params.seed)
    full = np.argmin(((flat[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    # un grupo es piso si su centro está cerca de algún modelo del piso
    floor_c = [i for i, c in enumerate(centers) if min(_mahal(c[None], g)[0] for g in models) < params.threshold]
    mask = ~np.isin(full, floor_c).reshape(h, w)
    return mask, None, seed


def _method_edges(rgb, params):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    v = float(np.median(gray))
    lo, hi = max(0, 0.66 * v), min(255, 1.33 * v)
    e = cv2.Canny(gray, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    e = cv2.morphologyEx(cv2.dilate(e, k), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.close_px + 7,) * 2))
    filled = _fill_holes(e > 0, max_frac=0.25)
    # quitar los trazos de borde que no encierran nada
    r = max(2, params.open_px + 2)
    filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    return filled


def _method_threshold(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = bw > 0
    return white if white.mean() <= 0.5 else ~white


# --------------------------------------------------------------------------- #
# Post-proceso
# --------------------------------------------------------------------------- #
def _fill_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    """Rellena regiones libres encerradas por obstáculos si son pequeñas."""
    free = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    h, w = mask.shape
    limit = max_frac * h * w
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches and area <= limit:
            out[lab == i] = True
    return out


def postprocess(mask: np.ndarray, params: PerceptionParams) -> tuple[np.ndarray, int]:
    m = mask.astype(np.uint8)
    if params.open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.open_px + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if params.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.close_px + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    out = m > 0
    if params.fill_holes:
        out = _fill_holes(out, params.max_hole_frac)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    min_area = params.min_area_frac * out.size
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    out = keep[lab]
    n_obj = int(keep.sum())
    if params.convex_hull and n_obj:
        hull = np.zeros(out.shape, np.uint8)
        cnts, _ = cv2.findContours(out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            cv2.fillPoly(hull, [cv2.convexHull(c)], 1)
        out = hull > 0
    if params.extra_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.extra_dilate_px + 1,) * 2)
        out = cv2.dilate(out.astype(np.uint8), k) > 0
    return out, n_obj


# --------------------------------------------------------------------------- #
# API principal
# --------------------------------------------------------------------------- #
def detect_obstacles(rgb: np.ndarray, params: PerceptionParams | None = None) -> PerceptionResult:
    """Detecta obstáculos en una imagen RGB ya rectificada/redimensionada."""
    params = params or PerceptionParams()
    if params.method not in METHODS:
        raise ValueError(f"Método desconocido: {params.method}")
    rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
    rng = np.random.default_rng(params.seed)
    t = {}
    t0 = time.perf_counter()
    img = rgb
    if params.denoise and params.denoise > 1:
        d = int(params.denoise)
        img = cv2.bilateralFilter(img, d, 30, d)
        img = cv2.medianBlur(img, 3)
    t["filtrado"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dist = None
    seed = None
    if params.method in ("piso", "combinado"):
        F = lab_features(img, params.shadow_weight)
        raw, dist, seed = _method_floor(F, params, rng, img)
        if params.method == "combinado":
            raw = raw | _method_edges(img, params)
    elif params.method == "kmeans":
        F = lab_features(img, params.shadow_weight)
        raw, dist, seed = _method_kmeans(F, params, rng)
    elif params.method == "bordes":
        raw = _method_edges(img, params)
    else:
        raw = _method_threshold(img)
    t["segmentacion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask, n_obj = postprocess(raw, params)
    t["morfologia"] = time.perf_counter() - t0
    return PerceptionResult(rgb, mask, dist, raw, seed, n_obj, t, params)


def process_photo(photo_rgb: np.ndarray, params: PerceptionParams | None = None, corners=None,
                  width_m: float = 5.0, height_m: float | None = None):
    """Foto completa -> (PerceptionResult, metros_por_píxel).

    Con ``corners`` (4 puntos en la foto original) se rectifica la perspectiva;
    ``width_m``/``height_m`` son las medidas reales de ese rectángulo.  Sin
    esquinas la foto se considera cenital y ``width_m`` es su ancho real.
    """
    params = params or PerceptionParams()
    if corners is not None:
        if height_m is None:
            raise ValueError("Con esquinas hay que indicar el alto real del rectángulo")
        img, mpp, _ = rectify(photo_rgb, corners, width_m, height_m, params.max_side)
    else:
        img = resize_max(photo_rgb, params.max_side)
        mpp = width_m / img.shape[1]
    return detect_obstacles(img, params), mpp


def to_environment(res: PerceptionResult, meters_per_pixel: float, name: str = "foto") -> Environment:
    env = Environment(res.mask.copy(), meters_per_pixel, name=name)
    env.meta["background"] = res.image
    env.meta["perception"] = res.params.to_dict()
    return env


# --------------------------------------------------------------------------- #
# Métricas contra verdad de terreno
# --------------------------------------------------------------------------- #
def mask_metrics(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 0) -> dict:
    """IoU, precisión, exhaustividad y F1 de la clase obstáculo.

    Con ``tolerance_px`` > 0 se ignoran los píxeles a esa distancia del borde
    real (los contornos exactos son ambiguos incluso para un humano).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    valid = np.ones_like(gt)
    if tolerance_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
        g = gt.astype(np.uint8)
        band = (cv2.dilate(g, k) - cv2.erode(g, k)) > 0
        valid = ~band
    tp = np.sum(pred & gt & valid)
    fp = np.sum(pred & ~gt & valid)
    fn = np.sum(~pred & gt & valid)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"iou": float(iou), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def object_recall(pred: np.ndarray, gt: np.ndarray, min_cover: float = 0.5) -> tuple[int, int]:
    """(objetos reales detectados, objetos reales) — detectado si ≥ min_cover de su área."""
    n, lab = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    found = 0
    for i in range(1, n):
        m = lab == i
        if pred[m].mean() >= min_cover:
            found += 1
    return found, n - 1
