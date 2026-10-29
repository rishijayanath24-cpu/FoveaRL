"""Recompute every number stated in the paper from the episode-level results and flag mismatches.

Point estimates are recomputed independently from eval2_main.npz / eval2_sens.npz (episode level);
interval bounds come from results/stats2.json (the bootstrap is the only source for them).
A claim passes if the stated value equals the computed value rounded to the stated precision."""
import json
import numpy as np
from sim import ALL_PROFILES, EXTRA_UNSEEN
from evaluate2 import KEYS

KI = {k: i for i, k in enumerate(KEYS)}
Z = np.load("results/eval2_main.npz")
S = np.load("results/eval2_sens.npz")
ST = json.load(open("results/stats2.json"))
R1 = json.load(open("results/summary.json"))["table"]
R2 = json.load(open("results/summary_r2.json"))["table"]
MAIN = list(ALL_PROFILES)
fails = 0
n = 0


def m(meth, prof, key, src=Z):            # overall mean = mean of agent means (equal episodes)
    return float(src[f"{meth}__{prof}"][:, :, KI[key]].mean())


def agent_means(meth, prof, key):
    return Z[f"{meth}__{prof}"][:, :, KI[key]].mean(1)


def chk(label, stated, computed, nd=0):
    global fails, n
    n += 1
    ok = abs(round(computed, nd) - stated) <= 10 ** (-nd) * 0.51 or abs(computed - stated) <= 10 ** (-nd) * 0.51
    if not ok:
        fails += 1
    print(f"{'PASS' if ok else 'FAIL'}  {label:62s} stated {stated:>8}  computed {computed:9.4f}")


def ci(key, lo, hi, nd=0):
    v = ST[key]
    chk(key + " lo", lo, v["lo"], nd)
    chk(key + " hi", hi, v["hi"], nd)


def rng(label, lo, hi, vals, nd=0):
    chk(label + " (min)", lo, min(vals), nd)
    chk(label + " (max)", hi, max(vals), nd)


print("== A. Table II vs episode-level data (stored summary_r2 vs eval2_main.npz) ==")
worst = 0.0
for p in MAIN:
    for meth in ("fixed_low", "fixed_mid", "fixed_high", "rule", "rl_qoe", "rl_task"):
        for k in ("success", "collisions", "quality", "bw_mbps", "ret"):
            worst = max(worst, abs(R2[p][meth][k][0] - m(meth, p, k)))
chk("max |table - recomputed| over all Table II cells", 0, worst, 6)

print("\n== B. Main comparison text ==")
rng("Fixed-Low success %", 0, 15, [100 * m("fixed_low", p, "success") for p in MAIN])
chk("Fixed-High stable success %", 98, 100 * m("fixed_high", "stable", "success"))
chk("Fixed-High congested success %", 0, 100 * m("fixed_high", "congested", "success"))
chk("Fixed-High congested late frames % (~a third)", 32, m("fixed_high", "congested", "late_pct"))
rng("Fixed-Mid success %", 88, 98, [100 * m("fixed_mid", p, "success") for p in MAIN])
for p, d, lo, hi in (("stable", 4.5, -3, 14), ("congested", 2.5, -9, 14), ("highjitter", -1, -11, 9),
                     ("intermittent", -10, -20, -2)):
    k = f"{p}|rl_qoe|vs_fixed_mid|success"
    chk(k + " diff (pp)", d, 100 * (m("rl_qoe", p, "success") - m("fixed_mid", p, "success")), 1)
    ci(k, lo, hi)
for p, d, lo, hi in (("congested", 0.067, 0.054, 0.081), ("highjitter", 0.085, 0.068, 0.102),
                     ("intermittent", 0.074, 0.064, 0.084)):
    k = f"{p}|rl_qoe|vs_fixed_mid|quality"
    chk(k + " diff", d, m("rl_qoe", p, "quality") - m("fixed_mid", p, "quality"), 3)
    ci(k, lo, hi, 3)
    chk(k + " holm < 0.01 (1=yes)", 1, float(ST[k]["p_holm"] < 0.01))
ci("stable|rl_qoe|vs_fixed_mid|quality", -0.011, 0.103, 3)
chk("QoE quality SD across agents, stable", 0.08, float(agent_means("rl_qoe", "stable", "quality").std(ddof=1)), 2)
chk("QoE |sent-rate diff| max over profiles < 0.3 (1=yes)", 1,
    float(max(abs(m("rl_qoe", p, "bw_mbps") - m("fixed_mid", p, "bw_mbps")) for p in MAIN) < 0.3))
chk("QoE sent-rate CIs all span zero (1=yes)", 1,
    float(all(ST[f"{p}|rl_qoe|vs_fixed_mid|bw_mbps"]["lo"] < 0 < ST[f"{p}|rl_qoe|vs_fixed_mid|bw_mbps"]["hi"] for p in MAIN)))
rng("QoE return gain vs Mid", 7, 14, [m("rl_qoe", p, "ret") - m("fixed_mid", p, "ret") for p in MAIN])
chk("QoE return gains all Holm < 0.02 (1=yes)", 1,
    float(all(ST[f"{p}|rl_qoe|vs_fixed_mid|ret"]["p_holm"] < 0.02 for p in MAIN)))
for p, d, lo, hi in (("congested", 20, 6, 36), ("intermittent", 15, 2, 29)):
    k = f"{p}|rl_qoe|vs_rule|success"
    chk(k + " diff", d, 100 * (m("rl_qoe", p, "success") - m("rule", p, "success")))
    ci(k, lo, hi)
for p, d, lo, hi in (("congested", -23.5, -35, -12.5), ("highjitter", -20, -32.5, -8),
                     ("intermittent", -26.5, -40, -15), ("stable", -6.5, -14.5, 1.5)):
    k = f"{p}|rl_task|vs_fixed_mid|success"
    chk(k + " diff", d, 100 * (m("rl_task", p, "success") - m("fixed_mid", p, "success")), 1)
    ci(k, lo, hi, 1)
rng("Task extra sent rate vs Mid (Mbps)", 0.34, 0.75, [m("rl_task", p, "bw_mbps") - m("fixed_mid", p, "bw_mbps") for p in MAIN], 2)
rng("Task quality gain vs Mid", 0.014, 0.049, [m("rl_task", p, "quality") - m("fixed_mid", p, "quality") for p in MAIN], 3)
rng("Task switches/min", 162, 186, [m("rl_task", p, "switch_pm") for p in MAIN])
rng("QoE switches/min", 13, 77, [m("rl_qoe", p, "switch_pm") for p in MAIN])
rng("QoE - Task success (pp)", 11, 26, [100 * (m("rl_qoe", p, "success") - m("rl_task", p, "success")) for p in MAIN])
v = ST["variance"]
rng("SD across agents, QoE", 2.5, 6.4, [v[f"rl_qoe|{p}"]["sd_across_agents"] for p in MAIN], 1)
rng("SD across agents, Task", 3.8, 10.1, [v[f"rl_task|{p}"]["sd_across_agents"] for p in MAIN], 1)
rng("within-agent episode SE, QoE", 2.0, 5.1, [v[f"rl_qoe|{p}"]["mean_within_agent_episode_se"] for p in MAIN], 1)
rng("within-agent episode SE, Task", 5.5, 7.6, [v[f"rl_task|{p}"]["mean_within_agent_episode_se"] for p in MAIN], 1)

print("\n== C. Ablation (Table III and text): pooled = equal weight per profile and per episode ==")


def pooled(meth, key):
    return float(np.concatenate([Z[f"{meth}__{p}"] for p in MAIN], axis=1)[:, :, KI[key]].mean())


chk("table: collision-only success %", 92, 100 * pooled("task_col", "success"))
chk("table: task-aware success %", 73, 100 * pooled("rl_task", "success"))
chk("table: ctrl-err-only success %", 76, 100 * pooled("task_cte", "success"))
chk("text: collision-only minus task-aware (pp), same pooling", 18.9,
    100 * (pooled("task_col", "success") - pooled("rl_task", "success")), 1)
chk("stats2 point estimate equals same pooled difference", 18.9, ST["ablation_diff|task_col|vs|rl_task"]["diff"], 1)
ci("ablation_diff|task_col|vs|rl_task", 13.3, 24.6, 1)
chk("ctrl-err-only minus task-aware (pp)", 3.1, 100 * (pooled("task_cte", "success") - pooled("rl_task", "success")), 1)
ci("ablation_diff|task_cte|vs|rl_task", -4.4, 10.9, 1)
chk("task+persist switches/min", 48, pooled("task_dwell", "switch_pm"))
chk("QoE+persist switches/min", 36, pooled("qoe_dwell", "switch_pm"))
chk("task+persist minus task (pp)", -5.4, 100 * (pooled("task_dwell", "success") - pooled("rl_task", "success")), 1)
ci("ablation_diff|task_dwell|vs|rl_task", -15.8, 5.4, 1)
chk("QoE+persist minus QoE (pp)", -12.9, 100 * (pooled("qoe_dwell", "success") - pooled("rl_qoe", "success")), 1)
ci("ablation_diff|qoe_dwell|vs|rl_qoe", -21.0, -4.8, 1)
abl = ["rl_qoe", "rl_task", "task_cte", "task_col", "task_dwell", "qoe_dwell"]
rng("latency component across variants", 0.19, 0.21, [pooled(a, "r_l") for a in abl], 2)
rng("bandwidth component across variants", 0.06, 0.10, [pooled(a, "r_b") for a in abl], 2)

print("\n== D. Held-out patterns (Table IV and text) ==")
unseen = ["intermittent"] + list(EXTRA_UNSEEN)
for meth, val in (("fixed_mid", 78), ("rl_qoe", 76), ("rule", 62), ("rl_task", 60), ("fixed_high", 35)):
    chk(f"mean over 6 held-out patterns, {meth}", val, float(np.mean([100 * m(meth, p, "success") for p in unseen])))
for p, q, mid in (("intermittent_mild", 94, 92), ("intermittent_severe", 85, 85), ("ramp", 95, 95),
                  ("square", 90, 92), ("intermittent", 88, 98)):
    chk(f"{p} QoE %", q, 100 * m("rl_qoe", p, "success"))
    chk(f"{p} Mid %", mid, 100 * m("fixed_mid", p, "success"))
chk("square rule %", 40, 100 * m("rule", "square", "success"))
chk("square QoE - rule", 50, 100 * (m("rl_qoe", "square", "success") - m("rule", "square", "success")))
ci("square|rl_qoe|vs_rule|success", 34, 66)
chk("severe rule %", 62, 100 * m("rule", "intermittent_severe", "success"))
chk("severe QoE - rule", 22.5, 100 * (m("rl_qoe", "intermittent_severe", "success") - m("rule", "intermittent_severe", "success")), 1)
ci("intermittent_severe|rl_qoe|vs_rule|success", 6.5, 40, 1)
chk("high-delay max success over methods <= 6", 6,
    max(100 * m(x, "highdelay", "success") for x in ("fixed_mid", "fixed_high", "rule", "rl_qoe", "rl_task")))

print("\n== E. Sensitivity (Table V and text) ==")


def spool(s, meth):
    return float(np.concatenate([S[f"{s}__{meth}__{p}"] for p in MAIN], axis=1)[:, :, KI["success"]].mean() * 100)


sens = ["default", "perc_low", "perc_high", "lat_low", "lat_high", "col_low", "col_high"]
rng("QoE - Task gap over settings", 8, 37, [spool(s, "rl_qoe") - spool(s, "rl_task") for s in sens])
for s, d, lo, hi in (("default", 28, 16, 42), ("perc_low", 8, 2, 14), ("perc_high", 37, 26, 46)):
    chk(f"{s} QoE-Task", d, spool(s, "rl_qoe") - spool(s, "rl_task"))
    chk(f"{s} lo", lo, ST[f"sens|{s}"]["qoe_minus_task"][1])
    chk(f"{s} hi", hi, ST[f"sens|{s}"]["qoe_minus_task"][2])
chk("all sensitivity CIs exclude zero (1=yes)", 1, float(all(ST[f"sens|{s}"]["qoe_minus_task"][1] > 0 for s in sens)))
chk("max |QoE - Mid| over settings <= 5", 5, max(abs(spool(s, "rl_qoe") - spool(s, "fixed_mid")) for s in sens))
chk("stronger-link QoE - Mid", 5, spool("perc_high", "rl_qoe") - spool("perc_high", "fixed_mid"))
chk("sens agents per setting (3)", 3, float(S["perc_low__rl_qoe__stable"].shape[0]))
chk("main agents (5)", 5, float(Z["rl_qoe__stable"].shape[0]))

print("\n== F. Training budget (text) ==")
for p, a, b in (("stable", 76, 86), ("congested", 59, 64), ("highjitter", 68, 72), ("intermittent", 63, 71)):
    chk(f"task success 700 ep, {p}", a, 100 * R1[p]["rl_task"]["success"][0])
    chk(f"task success 2000 ep, {p}", b, 100 * R2[p]["rl_task"]["success"][0])
for p, a, b in (("stable", 97, 97), ("congested", 89, 90), ("highjitter", 91, 92), ("intermittent", 89, 88)):
    chk(f"QoE success 700 ep, {p}", a, 100 * R1[p]["rl_qoe"]["success"][0])
    chk(f"QoE success 2000 ep, {p}", b, 100 * R2[p]["rl_qoe"]["success"][0])
rng("task switches 700 ep", 189, 202, [R1[p]["rl_task"]["switch_pm"][0] for p in MAIN])
cv = json.load(open("results/curves_r2.json"))
for v_, val in (("task", -37), ("qoe", 129)):
    C = np.array([cv[f"{v_}_{s}"] for s in range(5)])
    sm = np.array([np.convolve(c, np.ones(25) / 25, mode="valid") for c in C]).mean(0)
    chk(f"training plateau {v_}", val, float(sm[-200:].mean()))

print(f"\n{n - fails} of {n} claims PASS, {fails} FAIL")
