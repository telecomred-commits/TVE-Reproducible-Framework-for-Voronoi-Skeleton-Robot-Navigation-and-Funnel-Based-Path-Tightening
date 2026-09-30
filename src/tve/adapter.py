"""Adaptador del algoritmo mejorado TVE a la interfaz común de algoritmos."""
from __future__ import annotations

import time

from tve.tve import TVEConfig, TVEPlanner
from common.problem import AlgorithmResult, Problem


def run_tve(problem: Problem, generadores="auto", candidatas=4, holgura_extra_m=0.10, suavizado="arcos",
            radio_suavizado_m=1.0, portales="ambos", usar_cache=True):
    t0 = time.perf_counter()
    cfg = TVEConfig(robot_diameter=problem.robot_diameter, safety_margin=problem.safety_margin,
                    generators=generadores, candidates=int(candidatas), clearance_extra=float(holgura_extra_m),
                    smoothing=suavizado, turn_radius=float(radio_suavizado_m), portal_mode=portales)
    tp = TVEPlanner(problem.env, cfg)
    r = tp.plan(problem.start, problem.goal, use_cache=bool(usar_cache))
    stats = dict(r.stats)
    stats.update({f"t_{k}_ms": v * 1000 for k, v in r.timings.items()})
    viz = {"skeleton": [e.pts for e in tp.prep.graph.edges.values()] if tp.prep else [],
           "portals": r.portals, "taut": r.taut, "routes": [x["polyline"] for x in r.routes]}
    if not r.success:
        return AlgorithmResult("tve", "TVE (mejorado)", False, None, time.perf_counter() - t0, r.message, stats, viz)
    return AlgorithmResult("tve", "TVE (mejorado)", True, r.path, time.perf_counter() - t0, r.message, stats, viz)
