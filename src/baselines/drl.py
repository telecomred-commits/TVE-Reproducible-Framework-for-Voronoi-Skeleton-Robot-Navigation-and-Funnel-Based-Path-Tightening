"""Aprendizaje por refuerzo profundo (DRL): Double DQN con arquitectura dueling.

El agente NO planifica: aprende una política π(a | o) que, a partir de una
observación LOCAL y EGOCÉNTRICA, elige uno de 8 movimientos.  Se entrena una
sola vez en cientos de mapas aleatorios y luego se usa en mapas nuevos sin
reentrenar (generalización).

Observación (336 valores):
* parche 11×11 de ocupación alrededor del robot (celdas de 0.25 m);
* parche 11×11 de celdas ya visitadas en el episodio (memoria de corto plazo
  que ayuda a salir de trampas);
* parche 9×9 grueso (bloques de 4×4 celdas, ~9 m) con la fracción ocupada;
* 8 rayos: distancia libre en cada dirección (hasta 10 celdas);
* dirección y distancia a la meta.

Recompensa: +10 al llegar, −0.3 al chocar, −0.01 por paso, penalización por
revisitar, y un término de avance (γΦ(s′) − Φ(s), con Φ = −0.2·distancia), que
no cambia la política óptima (Ng et al., 1999).

Algoritmo: Double DQN (van Hasselt et al., 2016) + dueling (Wang et al., 2016)
+ retornos de 3 pasos + currículo de distancia a la meta.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.problem import AlgorithmResult, Problem, cell_center, coarse_grid, point_cell

CELL_M = 0.25
FINE_R = 5          # parche fino 11×11
COARSE_F = 4        # bloques de 4×4 celdas
COARSE_R = 4        # parche grueso 9×9
RAY_MAX = 10
ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])
ACT_LEN = np.array([1, 1, 1, 1] + [math.sqrt(2)] * 4)
OBS_DIM = (2 * FINE_R + 1) ** 2 * 2 + (2 * COARSE_R + 1) ** 2 + 8 + 5
MODEL_PATH = Path(__file__).resolve().parent / "models" / "drl_dqn.pt"


# --------------------------------------------------------------------------- #
# Entorno de malla
# --------------------------------------------------------------------------- #
class GridNav:
    def __init__(self, blocked: np.ndarray, rng: np.random.Generator, max_steps: int | None = None):
        self.blocked = blocked.astype(bool)
        self.H, self.W = blocked.shape
        P = max(FINE_R, RAY_MAX) + 1
        self.P = P
        self.pad = np.ones((self.H + 2 * P, self.W + 2 * P), np.float32)
        self.pad[P:-P, P:-P] = self.blocked
        Hc, Wc = int(np.ceil(self.H / COARSE_F)), int(np.ceil(self.W / COARSE_F))
        tmp = np.ones((Hc * COARSE_F, Wc * COARSE_F), np.float32)
        tmp[:self.H, :self.W] = self.blocked
        coarse = tmp.reshape(Hc, COARSE_F, Wc, COARSE_F).mean(axis=(1, 3))
        self.cpad = np.ones((Hc + 2 * COARSE_R, Wc + 2 * COARSE_R), np.float32)
        self.cpad[COARSE_R:-COARSE_R, COARSE_R:-COARSE_R] = coarse
        self.rng = rng
        self.max_steps = max_steps or int(min(800, 4 * (self.H + self.W)))

    def reset(self, start, goal):
        self.pos = np.array(start, int)
        self.goal = np.array(goal, int)
        self.visits = np.zeros((self.H + 2 * self.P, self.W + 2 * self.P), np.float32)
        self.visits[self.pos[0] + self.P, self.pos[1] + self.P] = 1
        self.t = 0
        self.d = float(np.hypot(*(self.pos - self.goal)))
        return self.obs()

    def rays(self):
        r, c = self.pos + self.P
        out = np.empty(8, np.float32)
        for k, (dr, dc) in enumerate(ACTIONS):
            n = 0
            while n < RAY_MAX and self.pad[r + (n + 1) * dr, c + (n + 1) * dc] == 0:
                n += 1
            out[k] = n / RAY_MAX
        return out

    def obs(self):
        r, c = self.pos + self.P
        f = self.pad[r - FINE_R:r + FINE_R + 1, c - FINE_R:c + FINE_R + 1]
        v = np.minimum(self.visits[r - FINE_R:r + FINE_R + 1, c - FINE_R:c + FINE_R + 1], 3) / 3
        cr, cc = self.pos[0] // COARSE_F + COARSE_R, self.pos[1] // COARSE_F + COARSE_R
        g = self.cpad[cr - COARSE_R:cr + COARSE_R + 1, cc - COARSE_R:cc + COARSE_R + 1]
        dv = (self.goal - self.pos).astype(np.float32)
        dist = float(np.hypot(*dv))
        u = dv / dist if dist > 0 else np.zeros(2, np.float32)
        goal = np.array([np.clip(dv[0] / 20, -1, 1), np.clip(dv[1] / 20, -1, 1), u[0], u[1],
                         math.log1p(dist) / 5], np.float32)
        return np.concatenate([f.ravel(), v.ravel(), g.ravel(), self.rays(), goal]).astype(np.float32)

    def valid_moves(self):
        r, c = self.pos + self.P
        ok = np.zeros(8, bool)
        for k, (dr, dc) in enumerate(ACTIONS):
            if self.pad[r + dr, c + dc]:
                continue
            if dr and dc and (self.pad[r + dr, c] or self.pad[r, c + dc]):
                continue
            ok[k] = True
        return ok

    def step(self, a: int, gamma: float = 0.97):
        self.t += 1
        dr, dc = ACTIONS[a]
        r, c = self.pos + self.P
        blocked = self.pad[r + dr, c + dc] > 0 or (dr and dc and (self.pad[r + dr, c] or self.pad[r, c + dc]))
        rew = -0.01
        if blocked:
            rew -= 0.3
        else:
            self.pos = self.pos + (dr, dc)
            nr, nc = self.pos + self.P
            prev = self.visits[nr, nc]
            rew -= 0.05 * min(prev, 3)
            self.visits[nr, nc] += 1
        nd = float(np.hypot(*(self.pos - self.goal)))
        # Moldeado por potencial SIN descuento: con γ < 1, quedarse quieto lejos
        # de la meta daría recompensa positiva 0.2·d·(1−γ) y el agente aprendería
        # a chocar contra las paredes.
        rew += 0.2 * (self.d - nd)
        self.d = nd
        done = False
        if nd <= 1.0:
            rew += 10.0
            done = True
        trunc = self.t >= self.max_steps
        return self.obs(), rew, done, trunc


# --------------------------------------------------------------------------- #
# Red y agente
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_net():
    torch, nn = _torch()

    class Dueling(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
            self.v = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            self.a = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 8))

        def forward(self, x):
            h = self.body(x)
            a = self.a(h)
            return self.v(h) + a - a.mean(dim=1, keepdim=True)
    return Dueling()


_MODEL_CACHE: dict = {}


def load_model(path: Path | str = MODEL_PATH):
    torch, _ = _torch()
    path = Path(path)
    key = (str(path), path.stat().st_mtime if path.exists() else 0)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not path.exists():
        return None, {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (net, ckpt.get("meta", {}))
    return net, ckpt.get("meta", {})


def problem_grid(problem: Problem):
    cell = max(1, int(round(CELL_M / problem.mpp)))
    return coarse_grid(problem.occ, cell), cell


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    total_steps: int = 400_000
    n_envs: int = 32
    buffer: int = 150_000
    batch: int = 256
    lr: float = 3e-4
    gamma: float = 0.97
    nstep: int = 3
    tau: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.6
    updates_per_step: int = 4
    map_every: int = 8          # episodios por mapa antes de generar otro
    seed: int = 0


def random_training_grid(rng: np.random.Generator):
    """Mapa aleatorio para entrenar (escenarios variados, escala y robot variables)."""
    from common import scenarios as sc
    from common.cspace import configuration_space, robot_radius_px
    kinds = ["articulo_aleatorio", "formas", "bosque", "concavos", "habitaciones", "pasos_estrechos", "vacio",
             "laberinto"]
    probs = np.array([0.18, 0.18, 0.18, 0.14, 0.12, 0.08, 0.04, 0.08])
    kind = rng.choice(kinds, p=probs / probs.sum())
    size = int(rng.integers(160, 320))
    width = float(rng.uniform(8, 18))
    seed = int(rng.integers(0, 10 ** 6))
    if kind == "vacio":
        env = sc.empty(size, width_m=width)
    elif kind == "laberinto":
        env = sc.maze(seed=seed, size=size, width_m=width, cells=int(rng.integers(3, 6)))
    else:
        env = sc.GENERATORS[kind](seed=seed, size=size, width_m=width)
    robot = float(rng.uniform(0.35, 0.6))
    occ = configuration_space(env.obstacles, robot_radius_px(robot, 0, env.meters_per_pixel))
    cell = max(1, int(round(CELL_M / env.meters_per_pixel)))
    return coarse_grid(occ, cell), kind


def _sample_pair(blocked, rng, max_dist):
    free = np.argwhere(~blocked)
    if len(free) < 2:
        return None
    from cv2 import connectedComponents
    _, comp = connectedComponents((~blocked).astype(np.uint8), connectivity=8)
    for _ in range(50):
        s = free[rng.integers(len(free))]
        cand = free[(comp[free[:, 0], free[:, 1]] == comp[s[0], s[1]])]
        d = np.hypot(*(cand - s).T)
        ok = cand[(d >= 3) & (d <= max_dist)]
        if len(ok):
            return s, ok[rng.integers(len(ok))]
    return None


def train(cfg: TrainConfig = TrainConfig(), out_path: Path | str = MODEL_PATH, progress=None,
          grids: list | None = None, init_from: Path | str | None = None) -> dict:
    """Entrena el agente.  ``grids`` fija los mapas (ajuste fino a un entorno);
    ``progress(fraction, info)`` se llama periódicamente."""
    torch, nn = _torch()
    torch.set_num_threads(max(1, min(8, (__import__("os").cpu_count() or 2) // 2)))
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    net = build_net()
    if init_from is not None and Path(init_from).exists():
        net.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=False)["state_dict"])
    tgt = build_net()
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    N = cfg.buffer
    B_obs = np.zeros((N, OBS_DIM), np.float16)
    B_nobs = np.zeros((N, OBS_DIM), np.float16)
    B_a = np.zeros(N, np.int64)
    B_r = np.zeros(N, np.float32)
    B_d = np.zeros(N, np.float32)
    ptr = 0
    full = False

    def new_env(k):
        if grids:
            blocked = grids[rng.integers(len(grids))]
        else:
            blocked, _ = random_training_grid(rng)
        return GridNav(blocked, rng)

    envs = [new_env(k) for k in range(cfg.n_envs)]
    ep_count = [0] * cfg.n_envs
    nbuf = [[] for _ in range(cfg.n_envs)]
    obs = []
    max_dist = 8.0

    def reset(k):
        nonlocal max_dist
        e = envs[k]
        for _ in range(5):
            pair = _sample_pair(e.blocked, rng, max_dist)
            if pair is not None:
                return e.reset(*pair)
            envs[k] = e = new_env(k)
        return e.reset(*np.argwhere(~e.blocked)[:2])

    for k in range(cfg.n_envs):
        obs.append(reset(k))
    obs = np.stack(obs)
    steps = 0
    ep_rew = np.zeros(cfg.n_envs)
    hist = {"return": [], "success": [], "loss": [], "steps": []}
    recent_succ = []
    t0 = time.perf_counter()
    gpow = cfg.gamma ** np.arange(cfg.nstep + 1)
    while steps < cfg.total_steps:
        frac = steps / cfg.total_steps
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * frac / cfg.eps_frac)
        max_dist = 8.0 + 120.0 * min(1.0, frac / 0.5)          # currículo
        with torch.no_grad():
            q = net(torch.from_numpy(obs)).numpy()
        new_obs = np.empty_like(obs)
        for k, e in enumerate(envs):
            valid = e.valid_moves()
            if rng.random() < eps and valid.any():
                a = int(rng.choice(np.flatnonzero(valid)))
            else:
                a = int(np.argmax(q[k]))
            o2, r, done, trunc = e.step(a, cfg.gamma)
            ep_rew[k] += r
            nbuf[k].append((obs[k], a, r))
            end = done or trunc
            # Transiciones n-step: (o_t, a_t, Σ γ^i r_{t+i}, o_{t+n}, γ^n o 0 si terminó).
            # Con la ventana llena se emite la más antigua; al terminar el
            # episodio se vacían todas (truncado: se sigue haciendo bootstrap).
            while nbuf[k] and (len(nbuf[k]) >= cfg.nstep or end):
                seq = nbuf[k]
                R = sum(gpow[i] * t[2] for i, t in enumerate(seq))
                B_obs[ptr] = seq[0][0]
                B_a[ptr] = seq[0][1]
                B_r[ptr] = R
                B_nobs[ptr] = o2
                B_d[ptr] = 0.0 if done else gpow[len(seq)]
                ptr = (ptr + 1) % N
                full = full or ptr == 0
                nbuf[k].pop(0)
            if end:
                recent_succ.append(1.0 if done else 0.0)
                hist["return"].append(float(ep_rew[k]))
                ep_rew[k] = 0
                nbuf[k] = []
                ep_count[k] += 1
                if ep_count[k] % cfg.map_every == 0:
                    envs[k] = new_env(k)
                o2 = reset(k)
            new_obs[k] = o2
        obs = new_obs
        steps += cfg.n_envs
        size = N if full else ptr
        if size >= cfg.batch * 4:
            for _ in range(cfg.updates_per_step):
                idx = rng.integers(0, size, cfg.batch)
                o = torch.from_numpy(B_obs[idx].astype(np.float32))
                o2t = torch.from_numpy(B_nobs[idx].astype(np.float32))
                a = torch.from_numpy(B_a[idx])
                r = torch.from_numpy(B_r[idx])
                disc = torch.from_numpy(B_d[idx])
                with torch.no_grad():
                    a2 = net(o2t).argmax(dim=1, keepdim=True)            # Double DQN
                    y = r + disc * tgt(o2t).gather(1, a2).squeeze(1)
                qsa = net(o).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qsa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                with torch.no_grad():
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
            hist["loss"].append(float(loss.item()))
        if steps % (cfg.n_envs * 50) == 0:
            sr = float(np.mean(recent_succ[-200:])) if recent_succ else 0.0
            hist["success"].append(sr)
            hist["steps"].append(steps)
            if progress is not None:
                progress(steps / cfg.total_steps, {"steps": steps, "success": sr, "eps": eps,
                                                  "max_dist": max_dist, "time": time.perf_counter() - t0})
    meta = {"obs_dim": OBS_DIM, "cell_m": CELL_M, "train_steps": steps, "train_time_s": time.perf_counter() - t0,
            "history": hist, "config": cfg.__dict__}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "meta": meta}, out_path)
    _MODEL_CACHE.clear()
    return meta


# --------------------------------------------------------------------------- #
# Uso del agente entrenado
# --------------------------------------------------------------------------- #
def rollout(net, blocked, start, goal, rng, max_steps=None, explore_when_looping=True):
    import torch
    env = GridNav(blocked, rng, max_steps)
    o = env.reset(start, goal)
    cells = [tuple(env.pos)]
    done = False
    collisions = 0
    for _ in range(env.max_steps):
        with torch.no_grad():
            q = net(torch.from_numpy(o[None])).numpy()[0]
        valid = env.valid_moves()
        q = np.where(valid, q, -np.inf) if valid.any() else q
        a = int(np.argmax(q))
        r, c = env.pos + env.P
        if explore_when_looping and env.visits[r, c] >= 4 and valid.any() and rng.random() < 0.3:
            a = int(rng.choice(np.flatnonzero(valid)))
        o, rew, done, trunc = env.step(a)
        if tuple(env.pos) == cells[-1]:
            collisions += 1
        else:
            cells.append(tuple(env.pos))
        if done or trunc:
            break
    return cells, done, collisions


def run_drl(problem: Problem, semilla=0, max_pasos=0, modelo=str(MODEL_PATH)):
    t0 = time.perf_counter()
    import torch
    torch.set_num_threads(1)          # inferencia de a una observación: 1 hilo es lo más rápido
    net, meta = load_model(modelo)
    if net is None:
        return AlgorithmResult("drl", "DRL (DQN)", False, None, time.perf_counter() - t0,
                               "No hay modelo entrenado: ejecute simulations/train_drl.py o «Entrenar» en la GUI")
    grid, cell = problem_grid(problem)
    from baselines.grid_search import _attach
    s = _attach(problem, grid, cell, problem.start)
    g = _attach(problem, grid, cell, problem.goal)
    if s is None or g is None:
        return AlgorithmResult("drl", "DRL (DQN)", False, None, time.perf_counter() - t0,
                               "S o T no se pueden conectar a la malla del agente")
    rng = np.random.default_rng(semilla)
    cells, done, coll = rollout(net, grid, s, g, rng, max_steps=int(max_pasos) or None)
    stats = {"pasos": len(cells), "choques_intentados": coll, "celda_px": cell,
             "entrenado_pasos": meta.get("train_steps"), "revisitas": len(cells) - len(set(cells))}
    pts = np.asarray([cell_center(r, c, cell) for r, c in cells], float)
    viz = {"cells": np.asarray(cells), "cell": cell, "partial": np.vstack([problem.start, pts])}
    if not done:
        return AlgorithmResult("drl", "DRL (DQN)", False, None, time.perf_counter() - t0,
                               "El agente no alcanzó T (se quedó dando vueltas o se agotaron los pasos)", stats, viz)
    path = np.vstack([problem.start, pts, problem.goal])
    return AlgorithmResult("drl", "DRL (DQN)", True, path, time.perf_counter() - t0,
                           "El agente alcanzó T", stats, viz)
