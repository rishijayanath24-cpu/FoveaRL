import json, sys
s = json.load(open(f"results/summary{sys.argv[1] if len(sys.argv) > 1 else ''}.json"))
cols = ["success", "collisions", "lat_ms", "late_pct", "bw_mbps", "quality", "time_s", "switch_pm", "ret"]
for p, t in s["table"].items():
    print("\n==", p)
    print(f"{'method':18s}" + "".join(f"{c:>14s}" for c in cols))
    for m, v in t.items():
        print(f"{m:18s}" + "".join(f"{v[c][0]:8.2f}±{v[c][1]:4.2f}" for c in cols))
print("\nWilcoxon p-values (rl_task vs other)")
for k, v in s["stats"].items():
    print(f"{k:55s}{v:.4g}")
