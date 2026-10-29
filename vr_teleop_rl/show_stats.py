import json
r = json.load(open("results/stats2.json"))


def f(k):
    v = r[k]
    s = f"{v['diff']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] p={v['p']:.4g}"
    return s + (f" holm={v['p_holm']:.4g}" if "p_holm" in v else "")


for p in ["stable", "congested", "highjitter", "intermittent"]:
    for a in ["rl_qoe", "rl_task"]:
        for m in ["success", "quality", "bw_mbps", "ret"]:
            print(p, a, m, f(f"{p}|{a}|vs_fixed_mid|{m}"))
print()
for p in ["stable", "congested", "highjitter", "intermittent", "intermittent_mild",
          "intermittent_severe", "ramp", "square", "highdelay"]:
    print(p, "| qoe-task succ", f(f"{p}|rl_qoe|vs_rl_task|success"),
          "| qoe-rule succ", f(f"{p}|rl_qoe|vs_rule|success"))
print()
for p in ["intermittent_mild", "intermittent_severe", "ramp", "square", "highdelay"]:
    for a in ["rl_qoe", "rl_task"]:
        print(p, a, "vs mid succ", f(f"{p}|{a}|vs_fixed_mid|success"))
print()
for k in r:
    if k.startswith("ablation_diff"):
        print(k, f(k))
print()
for k, v in r["variance"].items():
    print(k, {a: round(b, 1) for a, b in v.items()})
print(r["unseen_mean_success"])
n_sig = sum(1 for k, v in r.items() if "p_holm" in v and v["p_holm"] < 0.05)
print("Holm-significant primary comparisons:", n_sig, "of 32")
for k, v in r.items():
    if "p_holm" in v and v["p_holm"] < 0.05:
        print("  sig:", k, round(v["diff"], 3))
