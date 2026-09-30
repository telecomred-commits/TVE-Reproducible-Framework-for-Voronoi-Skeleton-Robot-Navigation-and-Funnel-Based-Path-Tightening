"""Navegación con SLAM (localización y mapeo simultáneos) en un entorno DESCONOCIDO.

A diferencia de los demás algoritmos, el robot no conoce el mapa: sólo lleva un
LiDAR simulado y una odometría con ruido.  En cada paso:

1. **Predicción**: la pose estimada avanza con la odometría (que acumula error).
2. **Localización (scan matching correlativo, Olson 2009)**: se buscan pequeñas
   correcciones (Δx, Δy, Δθ) que mejor alinean el escaneo actual con el mapa
   construido hasta ahora.
3. **Mapeo**: mapa de ocupación con log-odds (Moravec y Elfes, 1985): las celdas
   atravesadas por cada haz se vuelven más «libres» y los impactos más
   «ocupadas».
4. **Planificación**: A* sobre el mapa estimado (lo desconocido se asume libre,
   política optimista); se replanifica cuando la ruta queda bloqueada por
   obstáculos recién descubiertos.

La ruta resultante es la trayectoria REAL recorrida: más larga que la óptima
porque el robot explora, pero obtenida sin mapa previo.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from common.cspace import dilate
from common.geometry import bilinear, polyline_free
from common.problem import AlgorithmResult, Problem, snap_free
from baselines.grid_search import grid_search


class _MapProblem:
    """Adaptador mínimo para reutilizar A* sobre el mapa estimado."""

    def __init__(self, occ, start, goal, mpp):
        self.occ = occ
        self.start = start
        self.goal = goal
        self._mpp = mpp

    @property
    def mpp(self):
        return self._mpp


def _raycast(occ_real, pose, angles, max_r):
    """Distancias reales de cada haz (muestreo a 0.5 px)."""
    x, y, th = pose
    h, w = occ_real.shape
    r = np.arange(0.5, max_r + 0.5, 0.5)
    a = th + angles
    xs = x + np.cos(a)[:, None] * r[None, :]
    ys = y + np.sin(a)[:, None] * r[None, :]
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    out = (xs < 0) | (ys < 0) | (xs > w - 1) | (ys > h - 1)
    hit = occ_real[yi, xi] | out
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), len(r))
    dist = np.where(first < len(r), r[np.minimum(first, len(r) - 1)], max_r)
    return dist, first < len(r)


def run_slam(problem: Problem, rango_lidar_m=3.0, haces=180, ruido_odometria=0.02, sesgo_escala=0.02,
             ruido_giro_deg=0.4, deriva_giro_deg=0.08, ruido_lidar_m=0.02, paso_m=0.15, max_pasos=1500,
             localizacion=True, tolerancia_meta_m=0.75, semilla=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(semilla)
    mpp = problem.mpp
    h, w = problem.shape
    real_obst = problem.env.obstacles.copy()
    real_obst[[0, -1], :] = True
    real_obst[:, [0, -1]] = True
    max_r = rango_lidar_m / mpp
    step = max(1.0, paso_m / mpp)
    angles = np.linspace(-np.pi, np.pi, int(haces), endpoint=False)
    sig_r = ruido_lidar_m / mpp
    R = problem.radius_px

    logodds = np.zeros((h, w), np.float32)
    L_OCC, L_FREE, L_MAX = 0.85, -0.4, 6.0
    # prior de la odometría: su incertidumbre por paso (con holgura)
    SIG_XY = max(0.5, 3.0 * (ruido_odometria + sesgo_escala) * step)
    SIG_TH = max(0.5, 2.0 * (ruido_giro_deg + deriva_giro_deg))
    tol_goal = max(tolerancia_meta_m / mpp, step)
    true_pose = np.array([problem.start[0], problem.start[1], 0.0])
    est_pose = true_pose.copy()
    odo_pose = true_pose[:2].copy()          # sólo odometría (referencia sin SLAM)
    odo_th = 0.0
    odo_err = []
    T = problem.goal
    true_path = [true_pose[:2].copy()]
    est_path = [est_pose[:2].copy()]
    snapshots = []
    path_plan = None
    replans = 0
    bumps = 0
    pose_err = []
    corrections = []
    success = False
    frame = np.zeros((h, w), bool)
    frame[[0, -1], :] = True
    frame[:, [0, -1]] = True

    def scan_points(pose, dist, hit):
        a = pose[2] + angles
        return np.stack([pose[0] + np.cos(a) * dist, pose[1] + np.sin(a) * dist], axis=1), hit

    def integrate(pose, dist, hit):
        # celdas libres a lo largo de cada haz, impacto ocupado
        a = pose[2] + angles
        rr = np.arange(0.0, max_r, 0.8)
        xs = pose[0] + np.cos(a)[:, None] * rr[None, :]
        ys = pose[1] + np.sin(a)[:, None] * rr[None, :]
        valid = rr[None, :] < (dist[:, None] - 1.0)
        xi = np.clip(np.round(xs[valid]).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys[valid]).astype(int), 0, h - 1)
        free_idx = np.unique(yi * w + xi)
        flat = logodds.ravel()
        flat[free_idx] += L_FREE
        # Superficies CONTINUAS: se unen los impactos de haces consecutivos que
        # están cerca (misma superficie).  Con sólo los puntos, el mapa sería una
        # línea punteada y el scan matching "alinearía puntos con puntos".
        exf = pose[0] + np.cos(a) * dist
        eyf = pose[1] + np.sin(a) * dist
        surf = np.zeros((h, w), np.uint8)
        n = len(a)
        for i in range(n):
            if not hit[i]:
                continue
            p = (int(round(exf[i])), int(round(eyf[i])))
            j = (i + 1) % n
            if hit[j] and math.hypot(exf[j] - exf[i], eyf[j] - eyf[i]) < max(4.0, 0.12 * dist[i]):
                cv2.line(surf, p, (int(round(exf[j])), int(round(eyf[j]))), 1, 1)
            else:
                surf[min(max(p[1], 0), h - 1), min(max(p[0], 0), w - 1)] = 1
        flat[np.flatnonzero(surf.ravel())] += L_OCC - L_FREE     # anula el "libre" del mismo haz
        np.clip(logodds, -L_MAX, L_MAX, out=logodds)

    def scan_match(pred, dist, hit):
        """Búsqueda correlativa en una ventana pequeña alrededor de la predicción."""
        """Modelo de campo de verosimilitud (Thrun, Burgard y Fox, 2005):
        log p(z | x) = Σ log(z_hit · exp(−d²/2σ²) + z_rand), con d la distancia de
        cada impacto al obstáculo más cercano del mapa, más un prior gaussiano
        de la odometría.  Búsqueda exhaustiva vectorizada en una ventana."""
        occ_now = logodds > 0.5
        if hit.sum() < 5 or occ_now.sum() < 20:
            return pred
        dt = cv2.distanceTransform((~occ_now).astype(np.uint8), cv2.DIST_L2, 3)
        loglik = np.log(0.9 * np.exp(-dt ** 2 / (2 * 1.5 ** 2)) + 0.1)
        a0 = angles[hit]
        d0 = dist[hit]

        def score(dx, dy, dth):
            """Log-verosimilitud (interpolación bilineal: objetivo suave) + prior."""
            dx = np.atleast_1d(np.asarray(dx, float))
            dy = np.atleast_1d(np.asarray(dy, float))
            a = pred[2] + dth + a0
            ex = pred[0] + np.cos(a) * d0
            ey = pred[1] + np.sin(a) * d0
            pts = np.stack([(ex[None, :] + dx[:, None]).ravel(), (ey[None, :] + dy[:, None]).ravel()], axis=1)
            ll = bilinear(loglik, pts).reshape(len(dx), -1).sum(axis=1)
            return ll - 0.5 * (dx ** 2 + dy ** 2) / SIG_XY ** 2 - 0.5 * (np.degrees(dth) / SIG_TH) ** 2

        # 1) búsqueda exhaustiva gruesa
        dths = np.radians(np.arange(-2.0, 2.01, 0.5))
        off = np.arange(-1.5, 1.51, 0.5)
        DX, DY = np.meshgrid(off, off)
        DX, DY = DX.ravel().astype(float), DY.ravel().astype(float)
        best, best_s = (0.0, 0.0, 0.0), -np.inf
        for dth in dths:
            sc = score(DX, DY, dth)
            k = int(np.argmax(sc))
            if sc[k] > best_s:
                best_s, best = sc[k], (DX[k], DY[k], dth)
        # 2) refinamiento sub-píxel / sub-grado (parábola por eje, 3 rondas)
        bx, by, bt = best
        for step_xy, step_th in ((0.5, np.radians(0.25)), (0.25, np.radians(0.12)), (0.12, np.radians(0.06))):
            for axis in range(3):
                d = step_xy if axis < 2 else step_th
                cand = [np.array([bx, by, bt]) + s * d * np.eye(3)[axis] for s in (-1, 0, 1)]
                f = [float(score(c[0], c[1], c[2])[0]) for c in cand]
                den = f[0] - 2 * f[1] + f[2]
                if den < 0:
                    shift = float(np.clip(0.5 * (f[0] - f[2]) / den, -1, 1))
                    bx, by, bt = np.array([bx, by, bt]) + shift * d * np.eye(3)[axis]
        return np.array([pred[0] + bx, pred[1] + by, pred[2] + bt])

    def est_cspace():
        occ = logodds > 0.9
        occ |= frame
        return dilate(occ, R) if R > 0 else occ

    dist, hit = _raycast(real_obst, true_pose, angles, max_r)
    dist = np.clip(dist + rng.normal(0, sig_r, dist.shape), 0.5, max_r)
    integrate(est_pose, dist, hit)
    n_steps = 0
    for n_steps in range(1, int(max_pasos) + 1):
        if np.hypot(*(est_pose[:2] - T)) <= step:          # el robot cree que llegó
            success = bool(np.hypot(*(true_pose[:2] - T)) <= tol_goal)
            break
        cs = est_cspace()
        need = path_plan is None or len(path_plan) < 1
        if not need:
            # ¿la ruta restante choca con obstáculos recién descubiertos?
            need = not polyline_free(cs, path_plan[: min(len(path_plan), 40)]) or \
                (len(path_plan) > 0 and not polyline_free(cs, np.vstack([est_pose[:2], path_plan[:1]]))
                 and not cs[int(round(est_pose[1])), int(round(est_pose[0]))])
        if need:
            s_free, _ = snap_free(cs, est_pose[:2]) if cs.any() else (est_pose[:2], 0)
            g_free, _ = snap_free(cs, T)
            res = grid_search(_MapProblem(cs, s_free, g_free, mpp), "astar", cell=3)
            replans += 1
            if not res.success:
                break
            path_plan = res.path[1:] if len(res.path) > 1 else res.path[-1:]
        # avanzar ``step`` a lo largo de la ruta planificada (en el marco estimado)
        target = path_plan[0]
        d = target - est_pose[:2]
        dn = np.hypot(*d)
        while dn < step and len(path_plan) > 1:
            path_plan = path_plan[1:]
            target = path_plan[0]
            d = target - est_pose[:2]
            dn = np.hypot(*d)
        s = min(step, dn)
        phi = math.atan2(d[1], d[0])                      # rumbo deseado (marco estimado)
        dth_cmd = (phi - est_pose[2] + math.pi) % (2 * math.pi) - math.pi
        # --- movimiento REAL: el giro y el avance tienen ruido y sesgo que se ACUMULAN
        true_pose[2] += dth_cmd + rng.normal(0, math.radians(ruido_giro_deg)) + math.radians(deriva_giro_deg)
        s_true = s * (1.0 + sesgo_escala + rng.normal(0, ruido_odometria))
        new_true = true_pose[:2] + s_true * np.array([math.cos(true_pose[2]), math.sin(true_pose[2])])
        if polyline_free(problem.occ, np.vstack([true_pose[:2], new_true])):
            true_pose[:2] = new_true
        else:
            bumps += 1
        # --- predicción con la odometría (lo que el robot CREE que hizo)
        pred = np.array([est_pose[0] + s * math.cos(phi), est_pose[1] + s * math.sin(phi), phi])
        odo_th += dth_cmd
        odo_pose = odo_pose + s * np.array([math.cos(odo_th), math.sin(odo_th)])
        dist, hit = _raycast(real_obst, true_pose, angles, max_r)
        dist = np.clip(dist + rng.normal(0, sig_r, dist.shape), 0.5, max_r)
        # el escaneo se mide en el marco real; en el marco del robot su orientación
        # coincide con la real salvo el error de rumbo, que el scan matching corrige
        est_pose = scan_match(pred, dist, hit) if localizacion else pred
        corrections.append(est_pose - pred)
        integrate(est_pose, dist, hit)
        if len(path_plan) and np.hypot(*(path_plan[0] - est_pose[:2])) < 1.0:
            path_plan = path_plan[1:]
        if len(path_plan) == 0:
            path_plan = None
        true_path.append(true_pose[:2].copy())
        est_path.append(est_pose[:2].copy())
        pose_err.append(float(np.hypot(*(true_pose[:2] - est_pose[:2]))))
        odo_err.append(float(np.hypot(*(true_pose[:2] - odo_pose))))
        if n_steps % 15 == 0:
            snapshots.append((len(true_path), (1.0 / (1.0 + np.exp(-logodds))).astype(np.float16)))
    prob = 1.0 / (1.0 + np.exp(-logodds))
    snapshots.append((len(true_path), prob.astype(np.float16)))
    true_path = np.asarray(true_path)
    if success and polyline_free(problem.occ, np.vstack([true_path[-1], T])):
        true_path = np.vstack([true_path, T])
    known = np.abs(logodds) > 0.5
    est_occ = logodds > 0.9
    truth = real_obst
    inter = (est_occ & truth & known).sum()
    union = ((est_occ | truth) & known).sum()
    free_space = ~problem.occ
    stats = {"pasos": n_steps, "replanificaciones": replans, "choques": bumps,
             "error_pose_rmse_m": float(np.sqrt(np.mean(np.square(pose_err)))) * mpp if pose_err else 0.0,
             "error_pose_max_m": float(np.max(pose_err)) * mpp if pose_err else 0.0,
             "error_solo_odometria_m": float(odo_err[-1]) * mpp if odo_err else 0.0,
             "error_final_slam_m": float(pose_err[-1]) * mpp if pose_err else 0.0,
             "iou_mapa": float(inter / union) if union else 1.0,
             "cobertura_pct": float(100 * (known & free_space).sum() / max(free_space.sum(), 1))}
    lidar_end = scan_points(true_pose, dist, hit)[0]
    viz = {"map": prob, "est_path": np.asarray(est_path), "snapshots": snapshots,
           "corrections": np.asarray(corrections),
           "lidar": np.stack([np.repeat(true_pose[None, :2], len(lidar_end), 0), lidar_end], axis=1),
           "true_path": true_path}
    if not success:
        return AlgorithmResult("slam", "SLAM", False, None, time.perf_counter() - t0,
                               "No llegó a T: " + (f"creyó llegar, pero su error de localización era de {stats['error_final_slam_m']:.2f} m" if n_steps < max_pasos and np.hypot(*(est_pose[:2] - T)) <= step else "sin ruta en el mapa descubierto o límite de pasos"), stats, viz)
    return AlgorithmResult("slam", "SLAM", True, true_path, time.perf_counter() - t0,
                           f"Llegó a T explorando un entorno desconocido ({replans} replanificaciones)", stats, viz)
