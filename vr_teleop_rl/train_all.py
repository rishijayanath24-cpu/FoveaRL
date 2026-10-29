"""Tune the rule baseline, then train DQN agents (5 seeds x {task-aware, QoE-only})."""
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from sim import TRAIN_PROFILES
from agents import RulePolicy, run_episode, train_dqn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
import sys

SEEDS = [0, 1, 2, 3, 4]
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 700
TAG = sys.argv[2] if len(sys.argv) > 2 else ""


def tune_rule():
    best, best_r = None, -1e9
    for rtt_hi, margin, qd_hi in itertools.product(
            (0.08, 0.10, 0.12, 0.15, 0.20, 0.30), (0.5, 0.7, 0.85, 1.0), (0.05, 0.08, 0.12, 0.20)):
        p = RulePolicy(rtt_hi, margin, qd_hi)
        rets = [run_episode(p, prof, 5000 + i)["ret"]
                for prof in TRAIN_PROFILES for i in range(20)]   # training seeds only
        r = float(np.mean(rets))
        if r > best_r:
            best, best_r = (rtt_hi, margin, qd_hi), r
    return {"rtt_hi": best[0], "margin": best[1], "qd_hi": best[2], "train_return": best_r}


def job(args):
    variant, seed = args
    w_c = 1.0 if variant == "task" else 0.0
    t = time.time()
    sd, curve = train_dqn(seed, w_c, episodes=EPISODES)
    torch.save(sd, os.path.join(OUT, f"dqn_{variant}_{seed}{TAG}.pt"))
    return variant, seed, curve, time.time() - t


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rp = os.path.join(OUT, "rule_params.json")
    if os.path.exists(rp):
        rule = json.load(open(rp))
    else:
        rule = tune_rule()
        json.dump(rule, open(rp, "w"), indent=2)
    print("rule params:", rule, f"({time.time() - t0:.0f}s)", flush=True)
    jobs = [(v, s) for v in ("task", "qoe") for s in SEEDS]
    curves = {}
    with Pool(10) as pool:
        for variant, seed, curve, dt in pool.imap_unordered(job, jobs):
            curves[f"{variant}_{seed}"] = curve
            print(f"done {variant} seed {seed} in {dt:.0f}s; last-50 mean return "
                  f"{np.mean(curve[-50:]):.1f}", flush=True)
    json.dump(curves, open(os.path.join(OUT, f"curves{TAG}.json"), "w"))
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
