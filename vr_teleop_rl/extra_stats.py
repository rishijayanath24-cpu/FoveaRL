"""Paired Wilcoxon tests (seed-averaged per-episode values) for rl_qoe and rl_task vs baselines, both rounds."""
import json
import numpy as np
from scipy.stats import wilcoxon
from sim import ALL_PROFILES

KEYS = ["lat_ms", "p95_ms", "late_pct", "bw_mbps", "quality", "path_err",
        "collisions", "success", "finished", "time_s", "switch_pm", "ret"]
out = {}
for tag in ("", "_r2"):
    z = np.load(f"results/raw_eval{tag}.npz")
    for p in ALL_PROFILES:
        for a in ("rl_qoe", "rl_task"):
            for b in ("fixed_mid", "rule"):
                for metric in ("ret", "success", "collisions", "quality", "bw_mbps"):
                    ki = KEYS.index(metric)
                    x = z[f"{a}__{p}"].mean(0)[:, ki]
                    y = z[f"{b}__{p}"].mean(0)[:, ki]
                    try:
                        pv = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
                    except ValueError:
                        pv = 1.0
                    out[f"{tag or '_r1'}|{p}|{a}_vs_{b}|{metric}"] = {
                        "diff": float(x.mean() - y.mean()), "p": pv}
json.dump(out, open("results/extra_stats.json", "w"), indent=1)
print(f"{'round':5s}{'profile':13s}{'metric':11s}{'qoe-mid':>10s}{'p':>9s}{'qoe-rule':>10s}{'p':>9s}")
for tag in ("_r1", "_r2"):
    for p in ALL_PROFILES:
        for m in ("ret", "success", "collisions", "quality", "bw_mbps"):
            a = out[f"{tag}|{p}|rl_qoe_vs_fixed_mid|{m}"]
            b = out[f"{tag}|{p}|rl_qoe_vs_rule|{m}"]
            print(f"{tag:5s}{p:13s}{m:11s}{a['diff']:10.3f}{a['p']:9.3g}{b['diff']:10.3f}{b['p']:9.3g}")
