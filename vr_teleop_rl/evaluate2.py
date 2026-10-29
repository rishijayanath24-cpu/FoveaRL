"""Extended evaluation: ablations, extra unseen networks, sensitivity settings, reward components."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

from sim import ALL_PROFILES, EXTRA_UNSEEN
from agents import FixedPolicy, RulePolicy, DQNPolicy, QNet, run_episode

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
N_EP, EP_SEED0 = 40, 10000
KEYS = ["lat_ms", "p95_ms", "late_pct", "bw_mbps", "quality", "path_err", "collisions", "success",
        "finished", "time_s", "switch_pm", "ret", "r_q", "r_l", "r_b", "r_c", "r_s"]
BASELINES = ["fixed_low", "fixed_mid", "fixed_high", "rule"]
MAIN_RL = ["rl_qoe", "rl_task"]
ABL = ["task_cte", "task_col", "task_dwell", "qoe_dwell"]
MAIN_PROFILES = list(ALL_PROFILES) + list(EXTRA_UNSEEN)
SENS = ["perc_low", "perc_high", "lat_low", "lat_high", "col_low", "col_high"]
SENS_CFG = {
    "default": {},
    "perc_low": {"miss_coef": 0.4, "noise_coef": 0.35},
    "perc_high": {"miss_coef": 1.2, "noise_coef": 1.05},
    "lat_low": {"w_l": 0.4}, "lat_high": {"w_l": 1.6},
    "col_low": {"col_pen": 2.0}, "col_high": {"col_pen": 10.0},
}
EVAL_CFG = {"task_dwell": {"dwell": 5}, "qoe_dwell": {"dwell": 5}}   # dwell is part of the controller


def build(method, seed, tag):
    if method.startswith("fixed"):
        return FixedPolicy({"fixed_low": 0, "fixed_mid": 1, "fixed_high": 2}[method])
    if method == "rule":
        p = json.load(open(os.path.join(OUT, "rule_params.json")))
        return RulePolicy(p["rtt_hi"], p["margin"], p["qd_hi"])
    torch.set_num_threads(1)
    name = {"rl_qoe": "qoe", "rl_task": "task"}.get(method, method)
    q = QNet()
    q.load_state_dict(torch.load(os.path.join(OUT, f"dqn_{name}_{seed}{tag}.pt")))
    q.eval()
    return DQNPolicy(q, method)


def task(args):
    method, seed, tag, profiles, cfg = args
    pol = build(method, seed, tag)
    res = {}
    for prof in profiles:
        res[prof] = [[run_episode(pol, prof, EP_SEED0 + i, cfg=cfg)[k] for k in KEYS]
                     for i in range(N_EP)]
    return method, seed, res


def collect(jobs):
    with Pool(14) as pool:
        out = pool.map(task, jobs)
    raw = {}
    for m, s, res in out:
        raw.setdefault(m, {})[s] = res
    return raw


def main(which):
    if which in ("main", "all"):
        jobs = [(m, 0, "", MAIN_PROFILES, None) for m in BASELINES]
        jobs += [(m, s, "_r2", MAIN_PROFILES, None) for m in MAIN_RL for s in range(5)]
        jobs += [(m, s, "_r2", MAIN_PROFILES, EVAL_CFG.get(m)) for m in ABL for s in range(5)]
        raw = collect(jobs)
        arrs = {}
        for m, d in raw.items():
            for p in MAIN_PROFILES:
                arrs[f"{m}__{p}"] = np.array([d[s][p] for s in sorted(d)])
        np.savez_compressed(os.path.join(OUT, "eval2_main.npz"), **arrs)
        print("main done", flush=True)
    if which in ("sens", "all"):
        path = os.path.join(OUT, "eval2_sens.npz")
        arrs = dict(np.load(path)) if os.path.exists(path) else {}
        only = sys.argv[2:]                       # optionally re-run only some settings
        for sname, cfg in SENS_CFG.items():
            if only and sname not in only:
                continue
            # 'default' uses the 700-episode agents (seeds 0-2) so every row has the same
            # training budget and number of agents
            tag = "" if sname == "default" else f"_sens_{sname}"
            seeds = range(3)
            jobs = [(m, 0, "", list(ALL_PROFILES), cfg) for m in ("fixed_mid", "fixed_high", "rule")]
            jobs += [(m, s, tag, list(ALL_PROFILES), cfg) for m in MAIN_RL for s in seeds]
            raw = collect(jobs)
            for m, d in raw.items():
                for p in ALL_PROFILES:
                    arrs[f"{sname}__{m}__{p}"] = np.array([d[s][p] for s in sorted(d)])
            print("sens", sname, "done", flush=True)
        np.savez_compressed(path, **arrs)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
