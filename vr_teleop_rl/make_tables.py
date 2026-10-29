"""Generate LaTeX tables directly from the evaluation JSON files (no hand-typed numbers)."""
import json
from sim import ALL_PROFILES

LABEL = {"fixed_low": "Fixed-Low", "fixed_mid": "Fixed-Mid", "fixed_high": "Fixed-High",
         "rule": "Rule-ABR", "rl_qoe": "DQN (QoE-only)", "rl_task": "DQN (task-aware)"}
PNAME = {"stable": "Stable", "congested": "Congested", "highjitter": "High-jitter",
         "intermittent": "Intermittent (unseen)"}
r1 = json.load(open("results/summary.json"))["table"]
r2 = json.load(open("results/summary_r2.json"))["table"]


def f(t, m, k, scale=1.0, nd=2):
    v, s = t[m][k]
    return f"{v * scale:.{nd}f}".replace("-", "$-$") + f"$\\pm${s * scale:.{nd}f}"


# ---- main table (round 2) -------------------------------------------------
SHORT = {"fixed_low": "Fixed-Low", "fixed_mid": "Fixed-Mid", "fixed_high": "Fixed-High",
         "rule": "Rule-ABR", "rl_qoe": "DQN-QoE", "rl_task": "DQN-Task"}


def mean_only(t, m, k, scale=1.0, nd=2):
    return f"{t[m][k][0] * scale:.{nd}f}".replace("-", "$-$")


rows = []
for p in ALL_PROFILES:
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{PNAME[p]}}}}} \\\\")
    for m in LABEL:
        t = r2[p]
        rows.append(" & ".join([
            "\\hspace{1mm}" + SHORT[m], f(t, m, "success", 100, 0),
            mean_only(t, m, "collisions", 1, 2), mean_only(t, m, "quality", 1, 2),
            mean_only(t, m, "bw_mbps", 1, 1) if m != "fixed_low" else mean_only(t, m, "bw_mbps", 1, 2),
            mean_only(t, m, "ret", 1, 0)]) + " \\\\")
    rows.append("\\hline")
main = ("\\begin{tabular}{lrrrrr}\n\\hline\n"
        "Method & Success (\\%) & Coll. & $Q$ & Mbps & Return \\\\\n\\hline\n"
        + "\n".join(rows[:-1]) + "\n\\hline\n\\end{tabular}\n")
open("results/table_main.tex", "w", encoding="utf-8").write(main)

# ---- training-budget table (round 1 vs round 2) ----------------------------
PSHORT = {"stable": "Stable", "congested": "Congested", "highjitter": "High-jit.",
          "intermittent": "Interm.$^{*}$"}
rows = []
for p in ALL_PROFILES:
    for m in ("rl_qoe", "rl_task"):
        rows.append(" & ".join([PSHORT[p], SHORT[m],
                                f(r1[p], m, "success", 100, 0), f(r2[p], m, "success", 100, 0),
                                f(r1[p], m, "switch_pm", 1, 0), f(r2[p], m, "switch_pm", 1, 0)]) + " \\\\")
rounds = ("\\begin{tabular}{llrrrr}\n\\hline\n"
          "Profile & Agent & \\multicolumn{2}{c}{Success (\\%)} & \\multicolumn{2}{c}{Switches/min} \\\\\n"
          " & & 700 & 2000 & 700 & 2000 \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
open("results/table_rounds.tex", "w", encoding="utf-8").write(rounds)
print(main)
print(rounds)
