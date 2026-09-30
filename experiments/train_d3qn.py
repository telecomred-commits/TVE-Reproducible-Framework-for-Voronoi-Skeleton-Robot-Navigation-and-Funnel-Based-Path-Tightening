"""Optional: retrains the D3QN baseline with the configuration used in the paper.

The experiments use the shipped model (src/baselines/models/drl_dqn.pt), so this step is
NOT required to reproduce the paper. The new model is written to results/d3qn/ and is
never used unless it is copied over the shipped one.

    python experiments/train_d3qn.py            # 10^6 steps (about 10 min on a desktop CPU)
    python experiments/train_d3qn.py --steps 50000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--out", default=str(ROOT / "results" / "d3qn" / "drl_dqn_retrained.pt"))
    a = ap.parse_args()
    from baselines import drl
    cfg = dict(load_config("d3qn_training")["config"])
    if a.steps:
        cfg["total_steps"] = a.steps
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def progress(f, info):
        print(f"  {100 * f:5.1f}%  steps={info['steps']:>8}  success(200 ep)={info['success']:.2f}  "
              f"eps={info['eps']:.2f}  {info['time']:.0f} s", flush=True)

    meta = drl.train(drl.TrainConfig(**cfg), out, progress=progress)
    print(f"trained in {time.perf_counter() - t0:.0f} s -> {out}")
    (out.with_suffix(".json")).write_text(json.dumps({k: v for k, v in meta.items() if k != "history"},
                                                     indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
