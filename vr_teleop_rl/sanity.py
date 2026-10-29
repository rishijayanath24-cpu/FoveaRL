import time
import numpy as np
from sim import ALL_PROFILES
from agents import FixedPolicy, RulePolicy, run_episode

pols = [FixedPolicy(0), FixedPolicy(1), FixedPolicy(2), RulePolicy()]
t = time.time()
for prof in ALL_PROFILES:
    for p in pols:
        ms = [run_episode(p, prof, 20000 + i) for i in range(15)]
        g = lambda k: np.mean([m[k] for m in ms])
        print(f"{prof:12s}{p.name:11s} lat={g('lat_ms'):6.1f} late%={g('late_pct'):5.1f} "
              f"bw={g('bw_mbps'):5.2f} Q={g('quality'):.3f} cte={g('path_err'):.3f} "
              f"col={g('collisions'):.2f} fin={g('finished'):.2f} succ={g('success'):.2f} "
              f"time={g('time_s'):5.1f} ret={g('ret'):7.1f}")
print('secs', time.time() - t)
