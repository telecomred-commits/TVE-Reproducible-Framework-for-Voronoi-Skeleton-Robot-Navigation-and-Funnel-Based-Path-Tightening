"""Adaptador del método del artículo (teselado + esqueleto + reducción + Bézier)
a la interfaz común de algoritmos, para compararlo con los demás."""
from __future__ import annotations

import time

import numpy as np

from baselines.wdt.config import PlannerConfig
from baselines.wdt.planner import Planner
from common.problem import AlgorithmResult, Problem


def run_tessellation(problem: Problem, optimizar=False, generadores="obstacles", suavizado="piecewise",
                     rutas="paper"):
    t0 = time.perf_counter()
    cfg = PlannerConfig(robot_diameter=problem.robot_diameter, safety_margin=problem.safety_margin,
                        optimize=bool(optimizar), generators=generadores, smoothing=suavizado,
                        route_method=rutas)
    p = Planner(problem.env, cfg)
    r = p.plan(problem.start, problem.goal)
    stats = {"nodos_esqueleto": len(p.base_graph.nodes) if p.base_graph else 0,
             "rutas_candidatas": len(r.routes), "ciclos_reduccion": r.reduction.cycles if r.reduction else 0}
    stats.update({f"t_{k}_ms": v * 1000 for k, v in r.timings.items() if k != "total"})
    edges = []
    if p.base_graph is not None:
        edges = [e.pts for e in p.base_graph.edges.values()]
    viz = {"skeleton": edges, "reduced": r.reduced_path, "route": r.best_route.polyline if r.best_route else None,
           "tess_labels": p.tess.labels if p.tess is not None else None}
    key = "teselado_opt" if optimizar else "teselado"
    name = "Teselado optimizado" if optimizar else "Teselado (artículo)"
    if not r.success:
        return AlgorithmResult(key, name, False, None, time.perf_counter() - t0, r.message, stats, viz)
    return AlgorithmResult(key, name, True, np.asarray(r.final_path, float), time.perf_counter() - t0,
                           r.message, stats, viz)
