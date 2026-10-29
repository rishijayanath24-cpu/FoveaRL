"""Regression test: the extended simulator must reproduce the stored main results exactly."""
import json
import numpy as np
import torch
from sim import ALL_PROFILES
from agents import FixedPolicy, RulePolicy, DQNPolicy, QNet, run_episode

summ = json.load(open("results/summary_r2.json"))["table"]
rp = json.load(open("results/rule_params.json"))
bad = 0
for prof in ALL_PROFILES:
    for name, pol in (("fixed_mid", FixedPolicy(1)), ("fixed_high", FixedPolicy(2)),
                      ("rule", RulePolicy(rp["rtt_hi"], rp["margin"], rp["qd_hi"]))):
        ms = [run_episode(pol, prof, 10000 + i) for i in range(40)]
        for k in ("success", "collisions", "ret", "bw_mbps"):
            new = float(np.mean([m[k] for m in ms]))
            old = summ[prof][name][k][0]
            if abs(new - old) > 1e-6:
                bad += 1
                print("MISMATCH", prof, name, k, new, old)
# one trained agent
q = QNet()
q.load_state_dict(torch.load("results/dqn_qoe_0_r2.pt"))
pol = DQNPolicy(q)
ms = [run_episode(pol, "congested", 10000 + i) for i in range(40)]
raw = np.load("results/raw_eval_r2.npz")["rl_qoe__congested"]       # (S, E, K)
KEYS = ["lat_ms", "p95_ms", "late_pct", "bw_mbps", "quality", "path_err",
        "collisions", "success", "finished", "time_s", "switch_pm", "ret"]
for k in ("success", "ret", "collisions"):
    new = float(np.mean([m[k] for m in ms]))
    old = float(raw[0][:, KEYS.index(k)].mean())
    if abs(new - old) > 1e-6:
        bad += 1
        print("MISMATCH dqn", k, new, old)
print("regression:", "OK (identical)" if bad == 0 else f"{bad} mismatches")
if bad:
    raise SystemExit(1)
