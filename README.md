# Reproducibility package — TVE: Voronoi tessellation skeleton and funnel path tightening

This package reconstructs **all the results** of the article

> O. Penagos, I. Ladino, C. R. Suárez. *Fast, Near-Optimal and Curvature-Aware Robot Navigation on
> Camera-Derived Maps Using a Voronoi Tessellation Skeleton and Funnel Path Tightening.*
> Fundación Universitaria Los Libertadores, Bogotá, Colombia.

It contains everything needed by a reviewer or a third party to **generate the scenes, run experiments
E1–E7, and reproduce every table, figure and number of the paper**: the implementation of TVE, the fourteen
baselines, the seven scene generators, the camera image-formation model, the image-processing pipeline, the
exact parameters and seeds, the statistical analysis, the raw results that produced the paper and the
scripts that rebuild the tables and figures.

## 1. Contents

```
README.md                  this file
requirements.txt           exact versions of the dependencies (pip)
environment.yml            the same environment for conda
paper_experiments.py       master script of the RUN step (E1-E7)
paper_figures.py           master script of the ANALYZE and FIGURES steps
verify.py                  compares a new execution with the raw results of the paper
src/
  tve/                     TVE (tve.py) and its adapter to the common planner interface (adapter.py)
  baselines/               the fourteen baselines (see Section 6)
    wdt/                   WDT, the original wavefront-dilation tessellation method (Ladino et al., 2020)
    models/drl_dqn.pt      trained D3QN policy used in the paper
  common/                  shared infrastructure: environment, configuration space, exact collision
                           test, tessellation data structures, skeleton graph, planning problem, metrics
  perception/pipeline.py   image processing: rectification, floor-model segmentation, morphology
  registry.py              common interface of the fifteen planners (names, defaults, execution)
scenes/
  generators.py            the seven scene types (office, warehouse, apartment, workshop, store, plaza,
                           hospital ward)
  generate.py              GENERATE step: builds every scene and exports S and T of every query
image_formation/model.py   camera image-formation and degradation model (floor textures, shading,
                           cast shadows, illumination gradient, vignetting, noise, blur, JPEG, perspective)
configs/                   every parameter exactly as used (see Section 5)
seeds/experiment_seeds.csv seeds of E1-E7 and the rule that generates S and T
seeds/queries_E*.csv       S and T of every query (produced by scenes/generate.py)
experiments/               one script per experiment: e1_comparison.py ... e7_homotopy.py,
                           _common.py (shared pipeline) and train_d3qn.py (optional retraining)
statistics/                nonparametric.py (Friedman, Wilcoxon, Holm, bootstrap, effect size) and
                           analyze.py (ANALYZE step)
figures/make_figures.py    FIGURES step (Figures 2-11); output in figures/output/
results/raw/               raw results of the executions that produced the paper (e1.csv ... e7.csv)
results/tables/            tables of the paper as CSV, LaTeX bodies, numbers quoted in the text,
                           statistics_summary.json
tests/                     tests of the statistics and of the determinism of the pipeline
```

## 2. Installation

Python **3.14.6** was used (Windows 11 x64, Intel Core i7-12700KF, 64 GB RAM).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

or, with conda: `conda env create -f environment.yml && conda activate tve-reproducibility`.

The dependencies are NumPy 2.3.5, OpenCV 5.0.0 (`opencv-python`), pandas 3.0.3, Matplotlib 3.11.1,
PyTorch 2.14.0 (CPU, only for the D3QN baseline) and pytest 9.1.1. **SciPy is not required**: the
statistical tests are implemented in `statistics/nonparametric.py`.

Check the installation (about 1 minute):

```bash
python -m pytest tests -q
```

The test `test_scene_and_queries_match_paper_raw_results` rebuilds one scene from its seed, plans the first
query with TVE and checks that the result equals the corresponding row of `results/raw/e1.csv`.

## 3. Reproducing the paper: generate → run → analyze → figures

All commands are run from the root of the package.

### 3.1 Tables, figures and numbers from the raw results (no planner is executed, ~2 min)

```bash
python paper_figures.py
```

This regenerates, from `results/raw/`, every table of Section 4 of the paper (`results/tables/*.csv`, with
the LaTeX bodies in `results/tables/latex/`), every number quoted in the text
(`results/tables/latex/numbers.tex`, `results/tables/statistics_summary.json`) and Figures 2–11
(`figures/output/`). The tables, the numbers and the figures produced in this way are identical to those of
the manuscript.

### 3.2 Full reconstruction from the seeds

| Step | Command | Output | Time* |
|---|---|---|---|
| 1. generate | `python scenes/generate.py` (add `--images` to save the images) | `seeds/queries_E*.csv`, `results/scenes/` | ~5 min |
| 2. run | `python paper_experiments.py all --workers 8` | `results/rerun/e1.csv` … `e7.csv` | 45–90 min |
| 3. analyze | `python statistics/analyze.py --raw results/rerun` | `results/tables/` | <1 min |
| 4. figures | `python figures/make_figures.py --raw results/rerun` | `figures/output/` | ~2 min |
| check | `python verify.py` | report of identical columns | <1 min |

\* with 8 worker processes on the machine described above. Steps 3 and 4 can be run together with
`python paper_figures.py --raw results/rerun`.

Every experiment can also be run on its own, e.g. `python experiments/e2_tve_vs_wdt.py --workers 8`. The
option `--quick` runs one layout per scene and level (about 10 minutes for E1–E7); a quick execution is a
subset of the full one and can be checked with `python verify.py --subset`.

New executions are written to `results/rerun/` and never overwrite `results/raw/`.

### 3.3 What is exactly reproducible

* **Geometric and topological results** — maps, images, detected obstacles, S and T, path lengths, length
  ratios, validity, clearances, turning radii, sharp turns, travel times of the robot models, safety against
  the ground truth, homotopy classes and image-processing metrics — are deterministic: all random generators
  are seeded (Section 4) and no planner uses a time budget. `verify.py` compares every non-timing column of a
  new execution with `results/raw/`: boolean and categorical columns must match exactly and numeric columns
  within a relative tolerance of 10⁻⁵, because vectorized NumPy reductions can differ in the last bits
  depending on memory alignment. The full re-execution of E1–E7 from this package reproduced all the runs;
  its report is in `results/verification.txt`.
* **Computation times** depend on the processor, the operating system and the load; they are not expected to
  coincide in milliseconds, but the comparative trends and the statistical conclusions should be recovered.
  In our re-execution on the same computer under a heavier load, absolute times were about four times larger,
  while the paired speed-up of TVE over WDT (1.73, 95% CI 1.68–1.79, against 1.82 in the paper), the share
  of queries in which TVE was faster (87.0% against 86.5%) and the Friedman ranks were preserved
  (`results/verification.txt`).
  Every process is limited to one thread (`OMP_NUM_THREADS=1`, `cv2.setNumThreads(1)`,
  `torch.set_num_threads(1)`).

## 4. Seeds and generation of S and T

`seeds/experiment_seeds.csv` lists, for every experiment, the scenes, levels, layout seeds, number of queries
per map and the strings that seed the random generators:

* scene and image: `numpy.random.default_rng(zlib.crc32(f"paper|{scene}|{level}|{seed}"))` — it draws the
  image conditions within the ranges of the level, renders the image and perturbs the four floor corners
  (±2 px, as marked by an operator);
* queries: `default_rng(zlib.crc32(f"q{n}|{scene}|{level}|{seed}"))` (E4: `q4|{scene}|{seed}`).

S and T are drawn uniformly in the free space of **both** the detected and the ground-truth configuration
spaces, in the same connected region, at least 45% of the map side apart (30% in E3); the first two
conditions guarantee a valid query for every planner and a meaningful safety check
(`experiments/_common.py::sample_queries`). `scenes/generate.py` writes the resulting coordinates to
`seeds/queries_E*.csv` (x = column, y = row of the detected map, whose scale is also given). The same
coordinates are stored in every row of a new execution (`sx, sy, tx, ty`). The query files list every query
that is drawn; E5 discards the queries in which the new obstacle disconnects S from T or the first TVE route
fails (84 drawn, 76 used) and E7 those in which the reference optimum is not found (252 drawn, 250 used).

## 5. Configurations

| File | Content |
|---|---|
| `configs/experiment.json` | robot diameter (0.45 m), R_min (0.8 m), map side (400 px), levels, query separation, safety threshold, planner list |
| `configs/tve.json` | `planner`: TVEConfig used in E2, E3, E5, E6, E7; `registry`: parameters of the common interface used in E1 and E4; `ablation_E6`; `candidates_E7` |
| `configs/baselines.json` | the parameters passed to each of the fourteen baselines, the WDT configuration used when it is called directly, and a glossary of the (Spanish) parameter identifiers |
| `configs/perception.json` | image-processing parameters |
| `configs/image_formation.json` | ranges of the image-formation model for the low, medium and high levels |
| `configs/scenes.json` | scene generators and floor sizes |
| `configs/d3qn_training.json` | training configuration of the D3QN baseline |

The configuration files were generated by introspection of the objects used in the experiments, and the
experiment scripts read them; they are therefore the parameters that produced the paper. Notes:

* The parameters of the baselines are the default values of their implementations, chosen during their
  development following settings commonly reported in the literature, fixed before the experiments,
  identical for every map and query, and not tuned on the evaluation maps. Hybrid A* uses the turning radius
  R_min of the vehicle.
* In E1 and E4, TVE was executed through the common interface, whose default fillet radius is 1.0 m (a target
  radius larger than R_min = 0.8 m); in E2, E3, E5, E6 and E7 it was executed with R = R_min. The Ackermann
  feasibility is always evaluated with R_min = 0.8 m.
* The illustrative Figures 2–4 use the defaults of the common interface (Figure 4: Hybrid A* with a 1.0 m
  turning radius).

## 6. Planners

| Paper name | Key | Implementation |
|---|---|---|
| TVE (proposed) | `tve` | `src/tve/tve.py` |
| WDT (Ladino et al., 2020) | `teselado` | `src/baselines/wdt/` |
| Dijkstra, A*, Theta* | `dijkstra`, `astar`, `theta` | `src/baselines/grid_search.py` |
| Visibility graph (reference optimum) | `visibility` | `src/baselines/visibility.py` |
| RRT, RRT*, PRM | `rrt`, `rrt_star`, `prm` | `src/baselines/sampling.py` |
| APF | `potential` | `src/baselines/potential.py` |
| Hybrid A* | `hybrid_astar` | `src/baselines/hybrid_astar.py` |
| ACO | `aco` | `src/baselines/aco.py` |
| Homotopy planner | `topological` | `src/baselines/topological.py` |
| SLAM-based | `slam` | `src/baselines/slam.py` |
| DRL (D3QN) | `drl` | `src/baselines/drl.py` + `src/baselines/models/drl_dqn.pt` |

All planners receive the same `Problem` (`src/common/problem.py`): the detected configuration space, S, T and
the scale; all paths are validated with the same exact segment–pixel collision test
(`src/common/geometry.py`) and measured with the same metrics (`src/common/metrics.py`). The registry also
contains `teselado_opt`, a variant of WDT that is not used in the paper. The D3QN policy can be retrained with
`python experiments/train_d3qn.py` (optional; the experiments use the shipped model).

## 7. Experiments, tables and figures

| Exp. | Script | Paper | Raw results |
|---|---|---|---|
| E1 | `experiments/e1_comparison.py` | Tables 5, 6, 7, 9; Figures 5, 6, 8, 9 | `results/raw/e1.csv` |
| E2 | `experiments/e2_tve_vs_wdt.py` | Table 8; Figure 7; paired tests of Section 4.3 | `results/raw/e2.csv` |
| E3 | `experiments/e3_multiquery.py` | Figure 10 | `results/raw/e3.csv` |
| E4 | `experiments/e4_scalability.py` | Figure 11 | `results/raw/e4.csv` |
| E5 | `experiments/e5_replanning.py` | Table 10 | `results/raw/e5.csv` |
| E6 | `experiments/e6_ablation.py` | Table 11 | `results/raw/e6.csv` |
| E7 | `experiments/e7_homotopy.py` | Table 12 | `results/raw/e7.csv` |
| — | `figures/make_figures.py` | Figures 2, 3, 4 (qualitative) | — |

Tables 1–4 of the paper are descriptive (related work, scenes, levels, planners); Figure 1 is the TikZ flow
diagram of the manuscript.

**Statistics** (`statistics/nonparametric.py`, applied in `statistics/analyze.py`): Friedman test with average
ranks over the planners with at least 95% of valid paths and the queries solved by all of them; Wilcoxon
signed-rank tests (normal approximation with tie and continuity corrections) of TVE against every planner,
with Holm correction and effect size r = z/√n; percentile bootstrap (5000 resamples, seed 0) for the 95% CI of
the median speed-up. All values are written to `results/tables/statistics_summary.json`.

**Columns of the raw files** (identifiers of the source code): `exito` success, `sin_colision` collision-free
in the detected configuration space, `longitud_m` length, `relacion_optima` length / visibility-graph length,
`tiempo_ms` computation time, `holgura_min_m`/`holgura_media_m` minimum/mean clearance, `giro_total_deg`
total turning, `giros_bruscos` turns above 45°, `radio_giro_min_m` minimum turning radius,
`tiempo_holonomico_s`/`tiempo_diferencial_s`/`tiempo_ackermann_s` travel times, `factible_ackermann`
feasible for R_min, `gt_safe`/`gt_penetration_px`/`gt_clearance_m` safety against the ground truth,
`t_*_ms` stage times (`teselado` tessellation, `esqueleto` skeleton, `conexion_ST` S/T connection, `rutas`
route search, `reduccion` midpoint reduction, `embudo` funnel and validation, `suavizado` smoothing).

## 8. Notes

* Levels are named `facil`, `medio`, `dificil` in the code and low, medium, high in the paper.
* Comments inside the core modules are partly in Spanish, the language of the authors; interfaces,
  configuration files and this documentation are in English.
* The camera images are synthesized from the ground-truth layouts so that collisions with the real obstacles
  can be measured pixel by pixel; a validation with captured photographs is planned as future work (see the
  Discussion of the paper).
