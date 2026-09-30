"""Orquestador del algoritmo completo.

    Entorno ──► dilatación D/2 ──► teselado por frentes de onda ──► esqueleto + nodos
         ──► conexión de S y T ──► rutas F1..F4 ──► reducción ──► suavizado Bézier

``Planner.prepare()`` calcula todo lo que depende solo del mapa y del robot
(se reutiliza cuando solo cambian S y T).  ``Planner.plan(S, T)`` ejecuta el
resto y devuelve un ``PlanResult`` con todos los resultados intermedios.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from baselines.wdt.config import PlannerConfig
from common.cspace import configuration_space, robot_radius_px
from common.environment import Environment
from common.geometry import (bilinear, curvature, densify, polyline_free, polyline_length,
                       simplify_collision_free, turning_angles)
from common.gridsearch import astar
from baselines.wdt.reduction import ReductionResult, reduce_route, tighten
from baselines.wdt.routing import Route, dijkstra_route, plan_route
from common.skeleton import (SkeletonGraph, attach_point, bridge_components, extract_skeleton,
                       free_components)
from common.smoothing import SmoothResult, smooth_global, smooth_none, smooth_piecewise
from common.tessellation import Tessellation, tessellate

import cv2


@dataclass
class PlanResult:
    success: bool
    message: str
    start: np.ndarray | None = None
    goal: np.ndarray | None = None
    graph: SkeletonGraph | None = None
    s_node: int | None = None
    t_node: int | None = None
    routes: list[Route] = field(default_factory=list)
    best_index: int = -1              # ruta finalmente usada
    paper_best_index: int = -1        # ruta que elige F4 del artículo (más corta sin reducir)
    route_reduced_lengths: list = field(default_factory=list)
    waypoints: np.ndarray | None = None
    reduction: ReductionResult | None = None
    smooth: SmoothResult | None = None
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False
    bridges: int = 0

    @property
    def best_route(self) -> Route | None:
        return self.routes[self.best_index] if 0 <= self.best_index < len(self.routes) else None

    @property
    def reduced_path(self) -> np.ndarray | None:
        return None if self.reduction is None else self.reduction.path

    @property
    def final_path(self) -> np.ndarray | None:
        if self.smooth is not None:
            return self.smooth.curve
        return self.reduced_path


class Planner:
    def __init__(self, env: Environment, cfg: PlannerConfig | None = None) -> None:
        self.env = env
        self.cfg = cfg or PlannerConfig()
        self.radius_px: int = 0
        self.occ: np.ndarray | None = None
        self.tess: Tessellation | None = None
        self.base_graph: SkeletonGraph | None = None
        self.free_comp: np.ndarray | None = None
        self.real_clearance: np.ndarray | None = None
        self.prep_timings: dict = {}
        self._prepared_key = None

    # ------------------------------------------------------------------ prep
    def _key(self):
        c = self.cfg
        return (hash(self.env.obstacles.tobytes()), self.env.obstacles.shape, c.robot_diameter, c.safety_margin,
                c.dilation_shape, c.node_merge_dist, self.env.meters_per_pixel, c.generators, c.segment_size)

    def prepare(self, force: bool = False) -> None:
        if not force and self._prepared_key == self._key():
            return
        c = self.cfg
        t0 = time.perf_counter()
        self.radius_px = robot_radius_px(c.robot_diameter, c.safety_margin, self.env.meters_per_pixel)
        self.occ = configuration_space(self.env.obstacles, self.radius_px, c.dilation_shape)
        t1 = time.perf_counter()
        self.tess = tessellate(self.occ, c.generators, c.segment_size / self.env.meters_per_pixel)
        t2 = time.perf_counter()
        self.base_graph = extract_skeleton(self.tess, c.node_merge_dist)
        t3 = time.perf_counter()
        self.free_comp = free_components(self.occ)
        real_free = (~self.env.obstacles).astype(np.uint8)
        real_free[[0, -1], :] = 0
        real_free[:, [0, -1]] = 0
        self.real_clearance = cv2.distanceTransform(real_free, cv2.DIST_L2, 5).astype(np.float32)
        self.prep_timings = {"dilatacion": t1 - t0, "teselado": t2 - t1, "esqueleto": t3 - t2}
        self._prepared_key = self._key()

    # ------------------------------------------------------------- helpers
    def snap_to_free(self, p_xy) -> tuple[np.ndarray, float]:
        """Punto libre más cercano (si ``p_xy`` cae en un obstáculo dilatado)."""
        p = np.asarray(p_xy, dtype=float)
        h, w = self.occ.shape
        r, c = int(round(p[1])), int(round(p[0]))
        r = min(max(r, 0), h - 1)
        c = min(max(c, 0), w - 1)
        if not self.occ[r, c]:
            return np.array([c, r], dtype=float), 0.0
        ys, xs = np.nonzero(~self.occ)
        if len(ys) == 0:
            raise ValueError("El espacio de configuración no tiene celdas libres")
        d = (xs - p[0]) ** 2 + (ys - p[1]) ** 2
        k = int(np.argmin(d))
        return np.array([xs[k], ys[k]], dtype=float), float(np.sqrt(d[k]))

    def path_clearance(self, pts: np.ndarray) -> tuple[float, float]:
        """(holgura mínima en C-space, distancia mínima a obstáculos reales) [px]."""
        pts = densify(np.asarray(pts, dtype=float), 0.5)
        cs = float(bilinear(self.tess.clearance, pts).min()) if len(pts) else 0.0
        real = float(bilinear(self.real_clearance, pts).min()) if len(pts) else 0.0
        return cs, real

    # ------------------------------------------------------------------ plan
    def plan(self, start_xy, goal_xy) -> PlanResult:
        self.prepare()
        c = self.cfg
        mpp = self.env.meters_per_pixel
        timings = dict(self.prep_timings)
        warnings: list[str] = []

        start, ds = self.snap_to_free(start_xy)
        goal, dg = self.snap_to_free(goal_xy)
        if ds > 0:
            warnings.append(f"El inicio estaba dentro de un obstáculo dilatado; se movió {ds * mpp:.2f} m")
        if dg > 0:
            warnings.append(f"La meta estaba dentro de un obstáculo dilatado; se movió {dg * mpp:.2f} m")
        res = PlanResult(False, "", start=start, goal=goal, timings=timings, warnings=warnings)

        fs = self.free_comp[int(start[1]), int(start[0])]
        fg = self.free_comp[int(goal[1]), int(goal[0])]
        if fs != fg:
            res.message = "No existe ruta: inicio y meta están en regiones libres desconectadas"
            return res

        # --- conexión de S y T al esqueleto ---------------------------------
        t0 = time.perf_counter()
        g = self.base_graph.copy()
        s_node, s_ok = attach_point(g, self.tess, start, "start", c.collision_step)
        t_node, t_ok = attach_point(g, self.tess, goal, "goal", c.collision_step)
        if not (s_ok and t_ok):
            warnings.append("S o T no tienen esqueleto accesible en su región; se agregan puentes")
        comp = g.components()
        if comp[s_node] != comp[t_node]:
            res.bridges = bridge_components(g, self.tess, s_node, t_node)
            if res.bridges:
                warnings.append(f"Esqueleto desconectado en esta región: se agregaron {res.bridges} puente(s)")
        timings["conexion_ST"] = time.perf_counter() - t0
        res.graph, res.s_node, res.t_node = g, s_node, t_node

        # --- generación de rutas (F1-F4) -------------------------------------
        t0 = time.perf_counter()
        if np.allclose(start, goal):
            routes, best = [Route([s_node], [], 0.0, np.vstack([start, goal]), "trivial")], 0
        else:
            routes, best = plan_route(g, s_node, t_node, c.route_method)
        if not routes:
            path = astar(self.occ, start, goal)
            if path is None:
                res.message = "No se encontró ruta"
                timings["rutas"] = time.perf_counter() - t0
                return res
            routes, best = [Route([], [], polyline_length(path), path, "A* (respaldo)")], 0
            res.used_fallback = True
            warnings.append("El grafo no conecta S y T; se usó A* en malla como ruta inicial")
        if c.optimize and routes and routes[0].edges and c.route_method == "paper":
            d = dijkstra_route(g, s_node, t_node)
            if d is not None and all(tuple(d.edges) != tuple(r.edges) for r in routes):
                routes.append(d)
        timings["rutas"] = time.perf_counter() - t0
        res.routes, res.best_index, res.paper_best_index = routes, best, best

        # --- reducción --------------------------------------------------------
        t0 = time.perf_counter()
        D_px = c.robot_diameter / mpp
        min_dist = c.reduction_min_dist if c.reduction_min_dist is not None else D_px

        def reduce_candidate(route: Route):
            wp = self._waypoints(g, route)
            return wp, reduce_route(self.occ, wp, min_dist, c.reduction_max_cycles, c.collision_step)

        def refine(red: ReductionResult):
            tight = tighten(self.occ, red.path)
            if polyline_length(tight) < polyline_length(red.path) - 1e-9:
                red.history.append(tight)
                red.path = tight
                red.refined = True

        if c.optimize and len(routes) > 1:
            # Optimización: F4 sobre las rutas YA reducidas; el tensado solo se
            # aplica a las candidatas que pueden ganar (a <10 % de la mejor).
            cands = [reduce_candidate(r) for r in routes]
            lens = np.array([polyline_length(red.path) for _, red in cands])
            for i in np.flatnonzero(lens <= 1.10 * lens.min() + 1e-9):
                refine(cands[i][1])
            res.route_reduced_lengths = [polyline_length(red.path) for _, red in cands]
            best = int(np.argmin(res.route_reduced_lengths))
            res.best_index = best
            res.waypoints, res.reduction = cands[best]
        else:
            res.waypoints, res.reduction = reduce_candidate(routes[best])
            if c.optimize:
                refine(res.reduction)
            res.route_reduced_lengths = [np.nan] * len(routes)
            res.route_reduced_lengths[best] = polyline_length(res.reduction.path)
        timings["reduccion"] = time.perf_counter() - t0

        # --- suavizado ---------------------------------------------------------
        t0 = time.perf_counter()
        P = res.reduction.path
        if c.smoothing == "piecewise":
            res.smooth = smooth_piecewise(self.occ, P, c.smoothing_tension, c.smoothing_weight_gain,
                                          c.smoothing_samples_per_px)
        elif c.smoothing == "global":
            res.smooth = smooth_global(self.occ, P, c.smoothing_weight_gain, c.smoothing_samples_per_px)
            if not res.smooth.valid:
                warnings.append("La Bézier global (grado n) colisiona; use el suavizado 'piecewise'")
        else:
            res.smooth = smooth_none(self.occ, P)
        timings["suavizado"] = time.perf_counter() - t0

        res.metrics = self._metrics(res)
        res.success = True
        res.message = "Ruta encontrada"
        if not res.smooth.valid:
            res.message = "Ruta encontrada (el suavizado presenta colisión)"
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return res

    def _waypoints(self, g: SkeletonGraph, route: Route) -> np.ndarray:
        """Nodos de la ruta + los puntos mínimos para que cada cuerda sea libre."""
        step = self.cfg.collision_step
        if route.edges:
            parts = []
            for i, eid in enumerate(route.edges):
                pts = g.oriented_pts(eid, route.nodes[i])
                sub = simplify_collision_free(self.occ, pts, step=step)
                parts.append(sub if i == 0 else sub[1:])
            return np.vstack(parts)
        return simplify_collision_free(self.occ, route.polyline, step=step)

    # --------------------------------------------------------------- metrics
    def _metrics(self, res: PlanResult) -> dict:
        mpp = self.env.meters_per_pixel
        route = res.best_route
        P = res.reduction.path
        curve = res.smooth.curve
        k = curvature(densify(curve, 2.0)) / mpp
        cs_min, real_min = self.path_clearance(curve)
        red_free = polyline_free(self.occ, P)
        g = res.graph
        return {
            "teselas": self.tess.n_tiles,
            "dilataciones": self.tess.iterations,
            "nodos_esqueleto": sum(1 for n in self.base_graph.nodes),
            "aristas_esqueleto": len(self.base_graph.edges),
            "rutas_candidatas": len(res.routes),
            "nodos_ruta_inicial": len(route.nodes),
            "long_ruta_inicial_m": route.length * mpp,
            "puntos_antes_reduccion": len(res.waypoints),
            "puntos_ruta_reducida": len(P),
            "ciclos_reduccion": res.reduction.cycles,
            "long_ruta_reducida_m": polyline_length(P) * mpp,
            "giro_total_reducida_deg": float(np.degrees(turning_angles(P).sum())),
            "long_ruta_suave_m": polyline_length(curve) * mpp,
            "curvatura_max_1_m": float(k.max()) if len(k) else 0.0,
            "holgura_min_cspace_m": cs_min * mpp,
            "dist_min_obstaculo_real_m": real_min * mpp,
            "radio_robot_m": self.radius_px * mpp,
            "reducida_sin_colision": bool(red_free),
            "suave_sin_colision": bool(res.smooth.valid),
            "puentes": res.bridges,
            "respaldo_astar": res.used_fallback,
            "nodos_grafo_con_ST": len(g.nodes) if g is not None else 0,
        }


def plan(env: Environment, start_xy, goal_xy, cfg: PlannerConfig | None = None) -> PlanResult:
    """Atajo funcional: ``plan(env, S, T)``."""
    return Planner(env, cfg).plan(start_xy, goal_xy)
