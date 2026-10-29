"""Simulation of VR teleoperation of a mobile robot over a variable network.

Pure-Python/NumPy abstraction (no Gazebo/ROS): a unicycle robot is driven by a
simulated operator who sees a delayed, quality-degraded video stream. The
streaming layer is controlled by (foveation radius, peripheral resolution,
bitrate). Perceptual quality is a *parametric proxy*, not measured SSIM/VMAF.
"""
import math
import random
from collections import deque

import numpy as np

DT = 0.05            # control/sim step [s]
K = 4                # sim steps per RL decision (0.2 s)
T_MAX = 50.0         # episode time limit [s]
N_STEPS = int(T_MAX / DT)

RADII = (0.15, 0.30, 0.50)     # foveal radius (screen half-width = 1)
PRES = (0.25, 0.5, 1.0)        # peripheral resolution scale
BRATES = (2.0, 6.0, 12.0)      # nominal full-frame bitrate [Mbps]
N_ACT = 27
STATE_DIM = 14

V_MAX, W_MAX = 1.2, 1.5
ROBOT_R, OBST_R = 0.3, 0.5
WP_RADIUS = 0.6

TRAIN_PROFILES = ("stable", "congested", "highjitter")
ALL_PROFILES = TRAIN_PROFILES + ("intermittent",)
# additional held-out network patterns (never used for training or tuning)
EXTRA_UNSEEN = ("intermittent_mild", "intermittent_severe", "ramp", "square", "highdelay")

W_Q, W_L, W_B, W_C, W_S = 1.0, 0.8, 0.4, 1.0, 0.1

# Simulator/reward constants that the sensitivity study varies. Defaults reproduce
# the main experiments exactly.
DEFAULT_CFG = {
    "miss_coef": 0.8,    # obstacle-miss probability = miss_coef * (1-Q)^1.5
    "noise_coef": 0.7,   # obstacle position noise = 0.04 + noise_coef * (1-Q)^2
    "w_l": W_L,          # latency / late-frame penalty weight
    "col_pen": 5.0,      # penalty per collision inside the control cost
    "w_cte": None,       # weight of cross-track term (None -> w_c)
    "w_col": None,       # weight of collision term   (None -> w_c)
    "dwell": 0,          # minimum number of decisions between setting changes
}


def decode(a):
    return a // 9, (a // 3) % 3, a % 3


def encode(i, j, k):
    return i * 9 + j * 3 + k


def codec_quality(b):
    return (1 - math.exp(-b / 3.0)) / (1 - math.exp(-4.0))


def sent_rate(i, j, k):
    r, p, b = RADII[i], PRES[j], BRATES[k]
    af = min(1.0, math.pi * r * r / 4.0)
    return b * (af + (1 - af) * p * p)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class Network:
    """Precomputes per-step one-way delay, jitter, loss and bandwidth."""

    def __init__(self, profile, rng):
        n = N_STEPS + 2
        u = rng.uniform
        self.d0 = np.zeros(n)
        self.jit = np.zeros(n)
        self.loss = np.zeros(n)
        self.W = np.zeros(n)
        if profile == "stable":
            self.d0[:] = 0.015 * u(0.8, 1.2)
            self.jit[:] = 0.003 * u(0.8, 1.2)
            self.loss[:] = 0.002
            self.W[:] = 20.0 * u(0.8, 1.2)
        elif profile == "congested":
            base, amp = 0.055 * u(0.8, 1.2), 0.02
            per, ph = u(6, 12), u(0, 2 * math.pi)
            t = np.arange(n) * DT
            self.d0[:] = base + amp * np.sin(2 * math.pi * t / per + ph)
            self.jit[:] = 0.010 * u(0.8, 1.2)
            self.loss[:] = 0.010
            i = 0
            while i < n:
                seg = int(u(3, 6) / DT)
                self.W[i:i + seg] = u(3.0, 9.0)
                i += seg
        elif profile == "highjitter":
            self.d0[:] = 0.030 * u(0.8, 1.2)
            self.jit[:] = 0.030 * u(0.8, 1.2)
            self.loss[:] = 0.010
            self.W[:] = 12.0 * u(0.8, 1.2) * (1 + 0.1 * np.array([rng.gauss(0, 1) for _ in range(n)]))
        elif profile in ("intermittent", "intermittent_mild", "intermittent_severe"):
            n_lo, n_hi, d_lo, d_hi = {"intermittent": (2, 4, 0.8, 2.0),
                                      "intermittent_mild": (1, 2, 0.5, 1.0),
                                      "intermittent_severe": (4, 6, 1.0, 3.0)}[profile]
            self.d0[:] = 0.025 * u(0.8, 1.2)
            self.jit[:] = 0.008 * u(0.8, 1.2)
            self.loss[:] = 0.005
            self.W[:] = 14.0 * u(0.8, 1.2)
            for _ in range(rng.randint(n_lo, n_hi)):
                s = int(u(5, T_MAX - 5) / DT)
                d = int(u(d_lo, d_hi) / DT)
                self.W[s:s + d] = 0.5
                self.loss[s:s + d] = 0.15
                self.d0[s:s + d] += 0.05
        elif profile == "ramp":          # capacity decays steadily over the episode
            self.d0[:] = 0.025 * u(0.8, 1.2)
            self.jit[:] = 0.005 * u(0.8, 1.2)
            self.loss[:] = 0.005
            self.W[:] = np.linspace(20.0, 2.0, n) * u(0.8, 1.2)
        elif profile == "square":        # capacity toggles between two levels
            self.d0[:] = 0.040 * u(0.8, 1.2)
            self.jit[:] = 0.015 * u(0.8, 1.2)
            self.loss[:] = 0.010
            hi_lvl, lo_lvl = 12.0 * u(0.8, 1.2), 2.0 * u(0.8, 1.2)
            i, hi = 0, True
            while i < n:
                seg = int(u(3, 5) / DT)
                self.W[i:i + seg] = hi_lvl if hi else lo_lvl
                i += seg
                hi = not hi
        elif profile == "highdelay":     # long, lossy path (satellite-like)
            self.d0[:] = 0.120 * u(0.8, 1.2)
            self.jit[:] = 0.020 * u(0.8, 1.2)
            self.loss[:] = 0.030
            self.W[:] = 10.0 * u(0.8, 1.2)
        else:
            raise ValueError(profile)
        self.W = np.clip(self.W, 0.3, None)


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


BASE_WPS = [(6, 0), (10, 4), (10, 10), (5, 13), (0, 10), (-3, 5), (0, 1)]


class TeleopEnv:
    def __init__(self, profile, seed, w_c=W_C, cfg=None):
        self.profile = profile
        self.seed = seed
        self.w_c = w_c
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.w_cte = w_c if self.cfg["w_cte"] is None else self.cfg["w_cte"]
        self.w_col = w_c if self.cfg["w_col"] is None else self.cfg["w_col"]
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        rng = self.rng
        self.net = Network(self.profile, rng)
        self.wps = [(x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5)) for x, y in BASE_WPS]
        self.start = (0.0, 0.0)
        pts = [self.start] + self.wps
        self.obst = []
        for a, b in zip(pts[:-1], pts[1:]):
            if rng.random() < 0.85:
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = math.hypot(dx, dy)
                off = rng.choice((-1, 1)) * rng.uniform(0.0, 0.6)
                self.obst.append((mx - dy / L * off, my + dx / L * off))
        self.x, self.y, self.th = 0.0, 0.0, 0.0
        self.v, self.w = 0.0, 0.0
        self.n = 0
        self.wp_i = 0
        self.pose_hist = [(0.0, 0.0, 0.0)]
        self.gaze = [0.0, 0.0]
        self.gaze_hist = [(0.0, 0.0)]
        self.gaze_tgt = [0.0, 0.0]
        self.fix_left = 0
        self.backlog = 0.0
        self.frame_age = 0.03
        self.last_cmd = (0.0, 0.0)
        self.pending = deque()
        self.inside = [False] * len(self.obst)
        self.collisions = 0
        self.success = False
        self.done = False
        self.act = encode(1, 1, 1)
        self.rtt_s, self.jit_s, self.loss_s = 0.05, 0.0, 0.0
        self.bw_est = float(self.net.W[0])
        self.qd = 0.0
        self.prev_lat = 0.03
        self.log = {k: [] for k in ("lat", "late", "S", "Q", "cte", "act", "W")}
        self.n_switch = 0
        self.since_switch = 10 ** 6
        self.ret = 0.0
        self.comp = {"q": 0.0, "l": 0.0, "b": 0.0, "c": 0.0, "s": 0.0, "n": 0}
        self.gspeed = 0.0
        self.cte = 0.0
        return self._state()

    # ------------------------------------------------------------- sub-models
    def _update_gaze(self):
        rng = self.rng
        self.fix_left -= 1
        if self.fix_left <= 0:
            m = rng.random()
            if m < 0.5:
                self.gaze_tgt = [rng.gauss(0, 0.12), rng.gauss(0, 0.12)]
            elif m < 0.75:
                self.gaze_tgt = [rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7)]
            else:
                self.gaze_tgt = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
            self.fix_left = int(rng.uniform(0.3, 1.0) / DT)
        px, py = self.gaze
        for c in (0, 1):
            self.gaze[c] += 0.6 * (self.gaze_tgt[c] - self.gaze[c]) + rng.gauss(0, 0.01)
        self.gspeed = math.hypot(self.gaze[0] - px, self.gaze[1] - py) / DT
        self.gaze_hist.append((self.gaze[0], self.gaze[1]))

    def _operator_command(self, Q, age, d_up):
        rng = self.rng
        idx = max(0, self.n - int(round(age / DT)))
        px, py, pth = self.pose_hist[idx]
        sig = 0.04 * (1 + 3 * (1 - Q))
        px += rng.gauss(0, sig)
        py += rng.gauss(0, sig)
        pth += rng.gauss(0, sig)
        lead = 0.5 * (age + d_up)
        px += self.last_cmd[0] * math.cos(pth) * lead
        py += self.last_cmd[1] * 0 + self.last_cmd[0] * math.sin(pth) * lead
        tx, ty = self.wps[min(self.wp_i, len(self.wps) - 1)]
        vx, vy = tx - px, ty - py
        d = math.hypot(vx, vy) + 1e-9
        vx, vy = vx / d, vy / d
        # obstacle perception degrades quadratically as visual quality drops:
        # position noise grows and the obstacle is intermittently not noticed
        sig_o = 0.04 + self.cfg["noise_coef"] * (1 - Q) ** 2
        p_miss = self.cfg["miss_coef"] * (1 - Q) ** 1.5
        steer, near = 0.0, 9.0
        for ox, oy in self.obst:
            if rng.random() < p_miss:
                continue
            dx, dy = ox + rng.gauss(0, sig_o) - px, oy + rng.gauss(0, sig_o) - py
            de = math.hypot(dx, dy)
            b = wrap(math.atan2(dy, dx) - pth)
            if de < 2.4 and abs(b) < 1.3:
                s = -1.0 if b >= 0 else 1.0
                steer += s * 1.5 * min(1.0, (2.4 - de) / 1.4)
                near = min(near, de)
        err = wrap(math.atan2(vy, vx) + steer - pth)
        slow = 1.0 if near > 1.6 else 0.5
        v_cmd = slow * V_MAX * max(0.2, math.cos(err)) / (1 + 4 * max(0.0, age - 0.08))
        w_cmd = max(-W_MAX, min(W_MAX, 2.0 * err))
        return v_cmd, w_cmd

    # ------------------------------------------------------------------- step
    def _substep(self):
        rng = self.rng
        n = self.n
        i, j, k = decode(self.act)
        r = RADII[i]
        self._update_gaze()
        d0, jit, loss, W = self.net.d0[n], self.net.jit[n], self.net.loss[n], self.net.W[n]

        S = sent_rate(i, j, k)
        self.backlog = max(0.0, self.backlog + (S - W) * DT)
        self.qd = self.backlog / W
        retx = rng.random() < loss
        lat = 0.015 + 0.002 * S + d0 + self.qd + rng.expovariate(1.0 / max(jit, 1e-6))
        if retx:
            lat += 2 * d0 + 0.02
        late = False
        if self.qd > 0.25 or lat > 0.30:
            late = True
            self.backlog = 0.05 * W
            self.frame_age = self.frame_age + DT
        else:
            self.frame_age = lat
        d_up = d0 + rng.expovariate(1.0 / max(jit, 1e-6))

        lag = self.frame_age + d0
        cidx = max(0, n - int(round(lag / DT)))
        cx, cy = self.gaze_hist[cidx]
        err = math.hypot(self.gaze[0] - cx, self.gaze[1] - cy)
        w_in = sigmoid((r - err) / 0.04)
        qf = codec_quality(BRATES[k])
        qp = qf * PRES[j] ** 0.6
        Q = w_in * qf + (1 - w_in) * qp

        v_cmd, w_cmd = self._operator_command(Q, self.frame_age, d_up)
        arr = n + max(1, int(round(d_up / DT)))
        self.pending.append((arr, v_cmd, w_cmd))
        self.last_cmd = (v_cmd, w_cmd)
        applied = None
        keep = deque()
        for it in self.pending:
            if it[0] <= n:
                applied = it
            else:
                keep.append(it)
        self.pending = keep
        if applied is not None:
            self.v_tgt, self.w_tgt = applied[1], applied[2]
        elif not hasattr(self, "v_tgt"):
            self.v_tgt, self.w_tgt = 0.0, 0.0
        self.v += max(-2.0 * DT, min(2.0 * DT, self.v_tgt - self.v))
        self.w = self.w_tgt
        self.x += self.v * math.cos(self.th) * DT
        self.y += self.v * math.sin(self.th) * DT
        self.th = wrap(self.th + self.w * DT)
        self.pose_hist.append((self.x, self.y, self.th))

        wx, wy = self.wps[self.wp_i]
        if math.hypot(wx - self.x, wy - self.y) < WP_RADIUS:
            self.wp_i += 1
            if self.wp_i >= len(self.wps):
                self.success = True
                self.done = True
        pts = [self.start] + self.wps
        a = pts[min(self.wp_i, len(self.wps) - 1)]
        b = pts[min(self.wp_i, len(self.wps) - 1) + 1]
        self.cte = seg_dist(self.x, self.y, a[0], a[1], b[0], b[1])
        newcol = 0
        for oi, (ox, oy) in enumerate(self.obst):
            hit = math.hypot(self.x - ox, self.y - oy) < ROBOT_R + OBST_R
            if hit and not self.inside[oi]:
                newcol += 1
                self.collisions += 1
                self.v = 0.0
            self.inside[oi] = hit

        # measured quantities for the agent
        self.rtt_s += 0.3 * ((lat if not late else self.frame_age) + d_up - self.rtt_s)
        self.jit_s += 0.3 * (abs(lat - self.prev_lat) - self.jit_s)
        self.prev_lat = lat
        self.loss_s += 0.1 * ((1.0 if (late or retx) else 0.0) - self.loss_s)
        self.bw_est += 0.2 * (W * (1 + 0.1 * rng.gauss(0, 1)) - self.bw_est)

        self.n += 1
        if self.n >= N_STEPS:
            self.done = True
        lg = self.log
        lg["lat"].append(self.frame_age)
        lg["late"].append(1.0 if late else 0.0)
        lg["S"].append(S)
        lg["Q"].append(Q)
        lg["cte"].append(self.cte)
        lg["act"].append(self.act)
        lg["W"].append(W)
        return Q, self.frame_age, late, S, self.cte, newcol

    def step(self, a):
        # action persistence: a requested change is ignored while the dwell lock holds
        if self.cfg["dwell"] > 0 and a != self.act and self.since_switch < self.cfg["dwell"]:
            a = self.act
        prev = decode(self.act)
        new = decode(a)
        sw = sum(1 for p, q in zip(prev, new) if p != q) / 3.0
        if sw > 0:
            self.n_switch += 1
            self.since_switch = 0
        else:
            self.since_switch += 1
        self.act = a
        Qs = Ls = Bs = Ccte = Ccol = 0.0
        cnt = 0
        for _ in range(K):
            if self.done:
                break
            Q, lat, late, S, cte, col = self._substep()
            Qs += Q
            Ls += min(lat / 0.30, 1.5) + (1.0 if late else 0.0)
            Bs += S / 12.0
            Ccte += min(cte / 0.5, 2.0)
            Ccol += self.cfg["col_pen"] * col
            cnt += 1
        cnt = max(cnt, 1)
        w_l = self.cfg["w_l"]
        R = (W_Q * Qs / cnt - w_l * Ls / cnt - W_B * Bs / cnt
             - (self.w_cte * Ccte + self.w_col * Ccol) / cnt - W_S * sw)
        self.ret += R
        c = self.comp                      # reward components (unit control weight)
        c["q"] += W_Q * Qs / cnt
        c["l"] += w_l * Ls / cnt
        c["b"] += W_B * Bs / cnt
        c["c"] += (Ccte + Ccol) / cnt
        c["s"] += W_S * sw
        c["n"] += 1
        return self._state(), R, self.done

    # ------------------------------------------------------------------ state
    def raw(self):
        return {"rtt": self.rtt_s, "jit": self.jit_s, "loss": self.loss_s,
                "bw": self.bw_est, "qd": self.qd, "act": self.act}

    def _state(self):
        i, j, k = decode(self.act)
        gx, gy = self.gaze
        s = [self.rtt_s / 0.3, self.jit_s / 0.1, self.loss_s / 0.1, self.bw_est / 20.0,
             gx, gy, min(self.gspeed / 10.0, 2.0), self.v / V_MAX, self.w / W_MAX,
             min(self.cte / 0.5, 2.0), self.qd / 0.25, i / 2.0, j / 2.0, k / 2.0]
        return np.asarray(s, dtype=np.float32)

    # ---------------------------------------------------------------- metrics
    def metrics(self):
        lg = self.log
        lat = np.asarray(lg["lat"]) * 1000.0
        T = max(self.n * DT, 1e-6)
        return {
            "lat_ms": float(lat.mean()),
            "p95_ms": float(np.percentile(lat, 95)),
            "late_pct": 100.0 * float(np.mean(lg["late"])),
            "bw_mbps": float(np.mean(lg["S"])),
            "quality": float(np.mean(lg["Q"])),
            "path_err": float(np.mean(lg["cte"])),
            "collisions": float(self.collisions),
            "success": 1.0 if (self.success and self.collisions == 0) else 0.0,
            "finished": 1.0 if self.success else 0.0,
            "time_s": T,
            "switch_pm": self.n_switch / (T / 60.0),
            "ret": float(self.ret),
            # mean per-decision reward components (all positive; return = q - l - b - c - s)
            "r_q": self.comp["q"] / max(self.comp["n"], 1),
            "r_l": self.comp["l"] / max(self.comp["n"], 1),
            "r_b": self.comp["b"] / max(self.comp["n"], 1),
            "r_c": self.comp["c"] / max(self.comp["n"], 1),
            "r_s": self.comp["s"] / max(self.comp["n"], 1),
        }
