"""Evaluate all methods on all profiles (held-out seeds), compute stats, make figures."""
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from sim import ALL_PROFILES, DT, decode, sent_rate
from agents import FixedPolicy, RulePolicy, DQNPolicy, QNet, run_episode

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SEEDS = [0, 1, 2, 3, 4]
N_EP = 40
EP_SEED0 = 10000
KEYS = ["lat_ms", "p95_ms", "late_pct", "bw_mbps", "quality", "path_err",
        "collisions", "success", "finished", "time_s", "switch_pm", "ret"]
METHODS = ["fixed_low", "fixed_mid", "fixed_high", "rule", "rl_qoe", "rl_task"]
LABEL = {"fixed_low": "Fixed-Low", "fixed_mid": "Fixed-Mid", "fixed_high": "Fixed-High",
         "rule": "Rule-ABR (tuned)", "rl_qoe": "DQN (QoE-only)", "rl_task": "DQN (task-aware)"}


def build(method, seed):
    if method.startswith("fixed"):
        return FixedPolicy({"fixed_low": 0, "fixed_mid": 1, "fixed_high": 2}[method])
    if method == "rule":
        p = json.load(open(os.path.join(OUT, "rule_params.json")))
        return RulePolicy(p["rtt_hi"], p["margin"], p["qd_hi"])
    torch.set_num_threads(1)
    q = QNet()
    q.load_state_dict(torch.load(os.path.join(OUT, f"dqn_{method[3:]}_{seed}{TAG}.pt")))
    q.eval()
    return DQNPolicy(q, method)


def task(args):
    method, seed = args
    pol = build(method, seed)
    res = {}
    for prof in ALL_PROFILES:
        res[prof] = [[run_episode(pol, prof, EP_SEED0 + i)[k] for k in KEYS] for i in range(N_EP)]
    return method, seed, res


def ci95(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    jobs = [(m, 0) for m in METHODS if not m.startswith("rl")] + \
           [(m, s) for m in ("rl_qoe", "rl_task") for s in SEEDS]
    with Pool(8) as pool:
        out = pool.map(task, jobs)
    raw = {}
    for m, s, res in out:
        raw.setdefault(m, {})[s] = res          # raw[m][seed][prof] -> [N_EP][len(KEYS)]
    # arrays: per method, per profile -> (n_seeds, N_EP, n_keys)
    A = {m: {p: np.array([raw[m][s][p] for s in sorted(raw[m])]) for p in ALL_PROFILES} for m in METHODS}

    summary = {"keys": KEYS, "table": {}, "stats": {}}
    for p in ALL_PROFILES:
        summary["table"][p] = {}
        for m in METHODS:
            a = A[m][p]                           # (S, E, K)
            seed_means = a.mean(1)                # (S, K)
            if a.shape[0] > 1:
                mean, spread = seed_means.mean(0), seed_means.std(0, ddof=1)
            else:
                mean, spread = seed_means[0], np.array([ci95(a[0][:, k]) for k in range(len(KEYS))])
            summary["table"][p][m] = {k: [float(mean[i]), float(spread[i])] for i, k in enumerate(KEYS)}
        # paired Wilcoxon on seed-averaged per-episode values
        for metric in ("ret", "collisions", "success"):
            ki = KEYS.index(metric)
            rl = A["rl_task"][p].mean(0)[:, ki]
            for other in ("rule", "fixed_mid", "rl_qoe"):
                ot = A[other][p].mean(0)[:, ki]
                try:
                    pv = float(wilcoxon(rl, ot).pvalue) if np.any(rl != ot) else 1.0
                except ValueError:
                    pv = 1.0
                summary["stats"][f"{p}|{metric}|rl_task_vs_{other}"] = pv
    json.dump(summary, open(os.path.join(OUT, f"summary{TAG}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"raw_eval{TAG}.npz"),
                        **{f"{m}__{p}": A[m][p] for m in METHODS for p in ALL_PROFILES})

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cols = {"fixed_low": "#9e9e9e", "fixed_mid": "#6baed6", "fixed_high": "#3182bd",
            "rule": "#e6a23c", "rl_qoe": "#a1d99b", "rl_task": "#d62728"}
    # Fig 2: success rate and bandwidth per profile
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.5))
    w = 0.13
    for mi, m in enumerate(METHODS):
        for ai, (key, ylab) in enumerate((("success", "Task success rate"), ("bw_mbps", "Mean sent bitrate (Mbps)"))):
            vals = [summary["table"][p][m][key][0] for p in ALL_PROFILES]
            errs = [summary["table"][p][m][key][1] for p in ALL_PROFILES]
            lo = [min(e, v) for e, v in zip(errs, vals)] if key == "success" else errs
            hi = [min(e, 1.0 - v) for e, v in zip(errs, vals)] if key == "success" else errs
            ax[ai].bar(np.arange(4) + (mi - 2.5) * w, vals, w, yerr=[lo, hi], color=cols[m],
                       label=LABEL[m], capsize=2.0, error_kw={"lw": 0.9})
            ax[ai].set_ylabel(ylab)
            ax[ai].set_xticks(range(4))
            ax[ai].set_xticklabels(["stable", "conges-\nted", "high-\njitter", "inter-\nmittent"],
                                   fontsize=8)
    ax[1].set_yscale("log")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=3, frameon=False, loc="upper center", columnspacing=1.0,
               handlelength=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(os.path.join(OUT, f"fig_results{TAG}.png"), dpi=300)
    plt.close(fig)

    # Fig 3: episode trace on the unseen intermittent profile
    fig, ax = plt.subplots(3, 1, figsize=(3.4, 3.5), sharex=True)
    seed_tr = EP_SEED0 + 3
    for m in ("fixed_high", "rule", "rl_qoe", "rl_task"):
        pol = build(m, 0)
        _, env = run_episode(pol, "intermittent", seed_tr, trace=True)
        t = np.arange(len(env.log["lat"])) * DT
        ax[0].plot(t, np.array(env.log["lat"]) * 1000, color=cols[m], lw=0.9, label=LABEL[m])
        ax[1].plot(t, env.log["S"], color=cols[m], lw=0.9)
        if m == "rl_qoe":
            ax[1].plot(t, env.log["W"], color="k", lw=0.6, ls="--", label="Available bandwidth")
    ax[0].set_ylabel("Age (ms)")
    ax[0].set_yscale("log")
    ax[1].set_ylabel("Sent (Mbps)")
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left", columnspacing=0.8,
                 handlelength=1.2)
    ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    pol = build("rl_qoe", 0)
    _, env = run_episode(pol, "intermittent", seed_tr, trace=True)
    t = np.arange(len(env.log["act"])) * DT
    # shade the network outages (available bandwidth at its 0.5 Mbps floor) on every panel
    out_mask = np.array(env.log["W"]) <= 0.6
    edges = np.flatnonzero(np.diff(np.concatenate(([0], out_mask.astype(int), [0]))))
    for s_i, e_i in zip(edges[::2], edges[1::2]):
        for a_ in ax:
            a_.axvspan(t[s_i], t[min(e_i, len(t) - 1)], color="0.85", zorder=0, lw=0)
    i_, j_, k_ = zip(*[decode(a) for a in env.log["act"]])
    ax[2].step(t, i_, lw=0.9, label="foveal radius lvl")
    ax[2].step(t, np.array(j_) + 0.04, lw=0.9, label="periph. res. lvl")
    ax[2].step(t, np.array(k_) + 0.08, lw=0.9, label="bitrate lvl")
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_ylim(-0.2, 3.1)
    ax[2].set_ylabel("Action level")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", columnspacing=0.6,
                 handlelength=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_trace{TAG}.png"), dpi=300)
    plt.close(fig)

    # Fig 4: training curves
    cv = json.load(open(os.path.join(OUT, f"curves{TAG}.json")))
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.6))
    for a, v, ttl in zip(ax, ("task", "qoe"), ("Task-aware reward", "QoE-only reward")):
        C = np.array([cv[f"{v}_{s}"] for s in SEEDS])
        sm = np.array([np.convolve(c, np.ones(25) / 25, mode="valid") for c in C])
        x = np.arange(sm.shape[1]) + 25
        a.plot(x, sm.mean(0), color=cols["rl_task" if v == "task" else "rl_qoe"])
        a.fill_between(x, sm.mean(0) - sm.std(0), sm.mean(0) + sm.std(0), alpha=0.25,
                       color=cols["rl_task" if v == "task" else "rl_qoe"])
        a.set_title(ttl, fontsize=9)
        a.set_xlabel("Training episode")
        a.set_ylabel("Return (25-ep. mean)")
        plateau = float(sm.mean(0)[-200:].mean())
        a.axhline(plateau, color="0.4", lw=0.7, ls=":")
        a.text(0.97, 0.08, f"final plateau $\\approx$ {plateau:.0f}", transform=a.transAxes,
               ha="right", fontsize=8, color="0.25")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_training{TAG}.png"), dpi=300)
    plt.close(fig)
    print("evaluation complete")


if __name__ == "__main__":
    main()
