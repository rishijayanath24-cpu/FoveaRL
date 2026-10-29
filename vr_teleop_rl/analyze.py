"""Hierarchical-bootstrap statistics, Holm correction, new tables and forest plot."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import ALL_PROFILES, EXTRA_UNSEEN
from evaluate2 import KEYS, ABL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KI = {k: i for i, k in enumerate(KEYS)}
NB = 5000
RNG = np.random.default_rng(12345)
PN = {"stable": "Stable", "congested": "Congested", "highjitter": "High-jitter",
      "intermittent": "Intermittent", "intermittent_mild": "Interm. (mild)",
      "intermittent_severe": "Interm. (severe)", "ramp": "Ramp-down", "square": "Square-wave",
      "highdelay": "High-delay"}


def hboot(A, B, metric, n=NB, scale=1.0):
    """Two-level bootstrap of mean(A) - mean(B).
    A: (agents_a, episodes, K), B: (agents_b, episodes, K). Agents are resampled independently
    for each method (they are independently trained); episodes are resampled jointly because
    every method is evaluated on the same environment seeds (paired design)."""
    a, b = A[:, :, KI[metric]], B[:, :, KI[metric]]
    Sa, E = a.shape
    Sb = b.shape[0]
    d = np.empty(n)
    for i in range(n):
        ia = RNG.integers(0, Sa, Sa)
        ib = RNG.integers(0, Sb, Sb)
        ie = RNG.integers(0, E, E)
        d[i] = a[ia][:, ie].mean() - b[ib][:, ie].mean()
    est = a.mean() - b.mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / n)
    return est * scale, lo * scale, hi * scale, p


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj


def pool(A):                       # concatenate profiles along the episode axis
    return np.concatenate(A, axis=1)


def main():
    Z = np.load(os.path.join(OUT, "eval2_main.npz"))
    get = lambda m, p: Z[f"{m}__{p}"]
    res = {}

    # ---------------- primary family: DQN variants vs Fixed-Mid on the 4 main profiles
    fam, labels = [], []
    for p in ALL_PROFILES:
        for a in ("rl_qoe", "rl_task"):
            for metric, sc in (("success", 100.0), ("quality", 1.0), ("bw_mbps", 1.0), ("ret", 1.0)):
                est, lo, hi, pv = hboot(get(a, p), get("fixed_mid", p), metric, scale=sc)
                fam.append(pv)
                labels.append((p, a, metric))
                res[f"{p}|{a}|vs_fixed_mid|{metric}"] = {"diff": est, "lo": lo, "hi": hi, "p": pv}
    adj = holm(np.array(fam))
    for (p, a, metric), pa in zip(labels, adj):
        res[f"{p}|{a}|vs_fixed_mid|{metric}"]["p_holm"] = float(pa)

    # ---------------- secondary comparisons (exploratory, uncorrected)
    for p in ALL_PROFILES + EXTRA_UNSEEN:
        for metric, sc in (("success", 100.0), ("ret", 1.0)):
            for a, b in (("rl_qoe", "rl_task"), ("rl_qoe", "rule"), ("rl_task", "rule")):
                est, lo, hi, pv = hboot(get(a, p), get(b, p), metric, scale=sc)
                res[f"{p}|{a}|vs_{b}|{metric}"] = {"diff": est, "lo": lo, "hi": hi, "p": pv}
    for p in EXTRA_UNSEEN:
        for a in ("rl_qoe", "rl_task"):
            est, lo, hi, pv = hboot(get(a, p), get("fixed_mid", p), "success", scale=100.0)
            res[f"{p}|{a}|vs_fixed_mid|success"] = {"diff": est, "lo": lo, "hi": hi, "p": pv}

    # ---------------- variance decomposition for DQN success (training seeds vs episodes)
    vd = {}
    for a in ("rl_qoe", "rl_task"):
        for p in ALL_PROFILES:
            x = get(a, p)[:, :, KI["success"]]                 # (agents, episodes)
            vd[f"{a}|{p}"] = {"sd_across_agents": float(x.mean(1).std(ddof=1) * 100),
                              "mean_within_agent_episode_se": float(
                                  np.mean(x.std(1, ddof=1) / np.sqrt(x.shape[1])) * 100)}
    res["variance"] = vd

    # ---------------- unseen-network table (success, %)
    meths = ["fixed_mid", "fixed_high", "rule", "rl_qoe", "rl_task"]
    short = {"fixed_mid": "Mid", "fixed_high": "High", "rule": "Rule", "rl_qoe": "QoE", "rl_task": "Task"}
    unseen = ["intermittent"] + list(EXTRA_UNSEEN)
    rows, means = [], {m: [] for m in meths}
    for p in unseen:
        vals = {m: float(get(m, p)[:, :, KI["success"]].mean() * 100) for m in meths}
        for m in meths:
            means[m].append(vals[m])
        rows.append(PN[p] + " & " + " & ".join(f"{vals[m]:.0f}" for m in meths) + " \\\\")
    rows.append("\\hline")
    rows.append("Mean & " + " & ".join(f"{np.mean(means[m]):.0f}" for m in meths) + " \\\\")
    res["unseen_mean_success"] = {m: float(np.mean(means[m])) for m in meths}
    tab = ("\\begin{tabular}{lrrrrr}\n\\hline\nUnseen profile & " + " & ".join(short[m] for m in meths)
           + " \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    open(os.path.join(OUT, "table_unseen.tex"), "w").write(tab)

    # ---------------- ablation table (pooled over the 4 main profiles)
    variants = [("rl_qoe", "QoE-only"), ("rl_task", "Task-aware"), ("task_cte", "Task: ctrl-err only"),
                ("task_col", "Task: collision only"), ("task_dwell", "Task + persist."),
                ("qoe_dwell", "QoE-only + persist.")]
    rows = []
    for m, lab in variants:
        P = pool([get(m, p) for p in ALL_PROFILES])             # (agents, 4E, K)
        per_agent = lambda k: P[:, :, KI[k]].mean(1)
        sd = lambda k: per_agent(k).std(ddof=1)
        mean = lambda k: per_agent(k).mean()
        res[f"ablation|{m}"] = {k: [float(mean(k)), float(sd(k))] for k in
                                ("success", "collisions", "switch_pm", "quality", "r_q", "r_l", "r_b",
                                 "r_c", "r_s", "ret", "late_pct")}
        rows.append(f"{lab} & {mean('success') * 100:.0f}$\\pm${sd('success') * 100:.0f} & "
                    f"{mean('collisions'):.2f} & {mean('switch_pm'):.0f}$\\pm${sd('switch_pm'):.0f} & "
                    f"{mean('r_q'):.2f} & {mean('r_l'):.2f} & {mean('r_b'):.2f} & {mean('r_c'):.2f} & "
                    f"{mean('ret'):.1f} \\\\")
    tab = ("\\begin{tabular}{lrrrrrrrr}\n\\hline\nVariant & Succ.(\\%) & Coll. & Sw/min & $+Q$ & $-L$ "
           "& $-B$ & $-C$ & Return \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    open(os.path.join(OUT, "table_ablation.tex"), "w").write(tab)
    # significance of dwell/ctrl variants vs task and vs qoe (pooled profiles)
    for a, b in (("task_dwell", "rl_task"), ("task_cte", "rl_task"), ("task_col", "rl_task"),
                 ("qoe_dwell", "rl_qoe"), ("rl_qoe", "rl_task"), ("task_dwell", "rl_qoe")):
        est, lo, hi, pv = hboot(pool([get(a, p) for p in ALL_PROFILES]),
                                pool([get(b, p) for p in ALL_PROFILES]), "success", scale=100.0)
        res[f"ablation_diff|{a}|vs|{b}"] = {"diff": est, "lo": lo, "hi": hi, "p": pv}

    # ---------------- sensitivity table
    S = np.load(os.path.join(OUT, "eval2_sens.npz"))
    sget = lambda s, m: pool([S[f"{s}__{m}__{p}"] for p in ALL_PROFILES])
    labels = {"default": "Default", "perc_low": "Weaker quality$\\to$perception link",
              "perc_high": "Stronger quality$\\to$perception link", "lat_low": "Latency weight 0.4 (default 0.8)",
              "lat_high": "Latency weight 1.6", "col_low": "Collision penalty 2 (default 5)",
              "col_high": "Collision penalty 10"}
    rows = []
    for s, lab in labels.items():
        mid = float(sget(s, "fixed_mid")[:, :, KI["success"]].mean() * 100)
        q = float(sget(s, "rl_qoe")[:, :, KI["success"]].mean() * 100)
        t = float(sget(s, "rl_task")[:, :, KI["success"]].mean() * 100)
        est, lo, hi, pv = hboot(sget(s, "rl_qoe"), sget(s, "rl_task"), "success", scale=100.0)
        est2, lo2, hi2, pv2 = hboot(sget(s, "rl_qoe"), sget(s, "fixed_mid"), "success", scale=100.0)
        res[f"sens|{s}"] = {"mid": mid, "qoe": q, "task": t,
                            "qoe_minus_task": [est, lo, hi, pv], "qoe_minus_mid": [est2, lo2, hi2, pv2]}
        rows.append(f"{lab} & {mid:.0f} & {q:.0f} & {t:.0f} & {est:.0f} [{lo:.0f}, {hi:.0f}] \\\\")
    tab = ("\\begin{tabular}{lrrrr}\n\\hline\nSetting & Mid & QoE & Task & QoE$-$Task [95\\% CI] \\\\\n\\hline\n"
           + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    open(os.path.join(OUT, "table_sens.tex"), "w").write(tab)

    json.dump(res, open(os.path.join(OUT, "stats2.json"), "w"), indent=1)

    # ---------------- forest plot: difference from Fixed-Mid with 95% CI
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(4, 1, figsize=(3.4, 4.9))
    short = {"stable": "Stable", "congested": "Congested", "highjitter": "High-jitter",
             "intermittent": "Interm."}
    spec = [("success", "$\\Delta$ success (pp)"), ("quality", "$\\Delta$ quality $Q$"),
            ("bw_mbps", "$\\Delta$ sent rate (Mbps)"), ("ret", "$\\Delta$ return")]
    colors = {"rl_qoe": "#2ca02c", "rl_task": "#d62728"}
    for ax, (metric, ttl) in zip(axs, spec):
        for pi, p in enumerate(ALL_PROFILES):
            for ai, a in enumerate(("rl_qoe", "rl_task")):
                r = res[f"{p}|{a}|vs_fixed_mid|{metric}"]
                y = pi + (0.18 if ai == 0 else -0.18)
                ax.plot([r["lo"], r["hi"]], [y, y], color=colors[a], lw=1.4)
                ax.plot(r["diff"], y, "o", color=colors[a], ms=3.5,
                        label=("DQN (QoE-only)" if a == "rl_qoe" else "DQN (task-aware)") if pi == 0 else None)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(4))
        ax.set_yticklabels([short[p] for p in ALL_PROFILES], fontsize=8)
        ax.set_ylim(-0.6, 3.6)
        ax.invert_yaxis()
        ax.set_xlabel(ttl)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, ncol=2, frameon=False, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig_forest.png"), dpi=300)
    print("analysis complete")


if __name__ == "__main__":
    main()
