"""Reward ablations (2000 ep, 5 seeds) and sensitivity runs (700 ep, 3 seeds)."""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from agents import train_dqn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# name -> (w_c, cfg)
ABLATION = {
    "task_cte": (1.0, {"w_col": 0.0}),       # control-error term only
    "task_col": (1.0, {"w_cte": 0.0}),       # collision term only
    "task_dwell": (1.0, {"dwell": 5}),       # task-aware + 1 s action persistence
    "qoe_dwell": (0.0, {"dwell": 5}),        # QoE-only + 1 s action persistence
}
ABL_SEEDS = [0, 1, 2, 3, 4]
ABL_EPISODES = 2000

SENS = {
    "perc_low": {"miss_coef": 0.4, "noise_coef": 0.35},
    "perc_high": {"miss_coef": 1.2, "noise_coef": 1.05},
    "lat_low": {"w_l": 0.4},
    "lat_high": {"w_l": 1.6},
    "col_low": {"col_pen": 2.0},
    "col_high": {"col_pen": 10.0},
}
SENS_SEEDS = [0, 1, 2]
SENS_EPISODES = 700


def job(args):
    fname, seed, w_c, cfg, episodes = args
    t = time.time()
    sd, curve = train_dqn(seed, w_c, episodes=episodes, cfg=cfg)
    torch.save(sd, os.path.join(OUT, fname))
    return fname, curve, time.time() - t


if __name__ == "__main__":
    jobs = []
    for name, (w_c, cfg) in ABLATION.items():
        for s in ABL_SEEDS:
            jobs.append((f"dqn_{name}_{s}_r2.pt", s, w_c, cfg, ABL_EPISODES))
    for sname, cfg in SENS.items():
        for variant, w_c in (("task", 1.0), ("qoe", 0.0)):
            for s in SENS_SEEDS:
                jobs.append((f"dqn_{variant}_{s}_sens_{sname}.pt", s, w_c, cfg, SENS_EPISODES))
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    curves = {}
    with Pool(15) as pool:
        for fname, curve, dt in pool.imap_unordered(job, jobs):
            curves[fname] = curve
            print(f"done {fname} in {dt:.0f}s (last-50 mean return {np.mean(curve[-50:]):.1f})", flush=True)
    json.dump(curves, open(os.path.join(OUT, "curves_extra.json"), "w"))
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
