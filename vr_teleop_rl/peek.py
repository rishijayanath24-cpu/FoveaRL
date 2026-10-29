import numpy as np
from sim import ALL_PROFILES, EXTRA_UNSEEN
from evaluate2 import KEYS
Z = np.load("results/eval2_main.npz")
ki = {k: i for i, k in enumerate(KEYS)}
methods = ["fixed_low", "fixed_mid", "fixed_high", "rule", "rl_qoe", "rl_task", "task_cte", "task_col", "task_dwell", "qoe_dwell"]
print("success % (mean over agents) per profile")
profs = list(ALL_PROFILES) + list(EXTRA_UNSEEN)
print(f"{'method':12s}" + "".join(f"{p[:10]:>11s}" for p in profs))
for m in methods:
    print(f"{m:12s}" + "".join(f"{Z[f'{m}__{p}'][:, :, ki['success']].mean() * 100:11.0f}" for p in profs))
print("\npooled over 4 main profiles: success, collisions, switches/min, r_q, r_l, r_b, r_c, ret")
for m in methods:
    P = np.concatenate([Z[f"{m}__{p}"] for p in ALL_PROFILES], axis=1)
    g = lambda k: P[:, :, ki[k]].mean()
    print(f"{m:12s} succ={g('success') * 100:5.1f} col={g('collisions'):.2f} sw={g('switch_pm'):6.1f} "
          f"q={g('r_q'):.3f} l={g('r_l'):.3f} b={g('r_b'):.3f} c={g('r_c'):.3f} ret={g('ret'):7.1f}")
