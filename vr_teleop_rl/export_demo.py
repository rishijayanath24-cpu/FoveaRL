"""Export real simulator episodes (trained agents + baselines) for the browser demo.

Episodes use the first four held-out evaluation seeds (10000-10003) of each profile; they are not
hand-picked. Everything shown in the demo is replayed from these recordings."""
import json
import math
import os

import torch

from sim import RADII, PRES, BRATES, DT
from agents import FixedPolicy, RulePolicy, DQNPolicy, QNet, run_episode

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "demo")
PROFILES = ["stable", "congested", "highjitter", "intermittent", "square", "ramp"]
SEEDS = [10000, 10001, 10002, 10003]
STRIDE = 2                                   # store every 2nd 50 ms step -> 10 Hz


def load_q(name, seed):
    torch.set_num_threads(1)
    q = QNet()
    q.load_state_dict(torch.load(os.path.join(RES, f"dqn_{name}_{seed}_r2.pt")))
    q.eval()
    return DQNPolicy(q)


rp = json.load(open(os.path.join(RES, "rule_params.json")))
METHODS = {
    "fixed_low": FixedPolicy(0),
    "fixed_mid": FixedPolicy(1),
    "fixed_high": FixedPolicy(2),
    "rule": RulePolicy(rp["rtt_hi"], rp["margin"], rp["qd_hi"]),
    "rl_qoe": load_q("qoe", 0),
    "rl_task": load_q("task", 0),
}


def r2(v, nd=2):
    return round(float(v), nd)


episodes = {}
for prof in PROFILES:
    for seed in SEEDS:
        key = f"{prof}|{seed}"
        ep = {"methods": {}}
        for name, pol in METHODS.items():
            m, env = run_episode(pol, prof, seed, trace=True)
            n = env.n
            idx = list(range(0, n, STRIDE))
            lg = env.log
            # collision events (same 0.8 m contact rule as the simulator)
            col, inside = [], [False] * len(env.obst)
            for k in range(n + 1):
                x, y, _ = env.pose_hist[k]
                for oi, (ox, oy) in enumerate(env.obst):
                    hit = math.hypot(x - ox, y - oy) < 0.8
                    if hit and not inside[oi]:
                        col.append(k // STRIDE)
                    inside[oi] = hit
            ep["methods"][name] = {
                "x": [r2(env.pose_hist[k][0]) for k in idx],
                "y": [r2(env.pose_hist[k][1]) for k in idx],
                "th": [r2(env.pose_hist[k][2]) for k in idx],
                "age": [int(lg["lat"][k] * 1000) for k in idx],
                "q": [r2(lg["Q"][k]) for k in idx],
                "s": [r2(lg["S"][k]) for k in idx],
                "act": [int(lg["act"][k]) for k in idx],
                "late": [int(lg["late"][k]) for k in idx],
                "gx": [r2(env.gaze_hist[k][0]) for k in idx],
                "gy": [r2(env.gaze_hist[k][1]) for k in idx],
                "col": col,
                "n": len(idx),
                "success": int(m["success"]), "finished": int(m["finished"]),
                "collisions": int(m["collisions"]), "time": r2(m["time_s"], 1),
                "bw": r2(m["bw_mbps"]), "quality": r2(m["quality"]),
                "sw": r2(m["switch_pm"], 0), "late_pct": r2(m["late_pct"], 1),
            }
            if name == "fixed_high":              # network is identical for all methods of an episode
                ep["w"] = [r2(lg["W"][k], 1) for k in idx]
                ep["wps"] = [[r2(a), r2(b)] for a, b in env.wps]
                ep["obst"] = [[r2(a), r2(b)] for a, b in env.obst]
        episodes[key] = ep
    print("done", prof, flush=True)

data = {
    "radii": list(RADII), "pres": list(PRES), "brates": list(BRATES), "dt": DT * STRIDE,
    "profiles": PROFILES, "seeds": SEEDS, "episodes": episodes,
    # headline results from the paper (2000-episode agents, 40 held-out episodes per profile)
    "paper": json.load(open(os.path.join(RES, "summary_r2.json")))["table"],
}
os.makedirs(OUT, exist_ok=True)
txt = "window.DEMO=" + json.dumps(data, separators=(",", ":")) + ";"
open(os.path.join(OUT, "data.js"), "w", encoding="utf-8").write(txt)
print("data.js", round(len(txt) / 1e6, 2), "MB")
