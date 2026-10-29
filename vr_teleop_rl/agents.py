"""Baseline controllers and the DQN agent."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from sim import N_ACT, STATE_DIM, encode, decode, sent_rate, TeleopEnv, ALL_PROFILES

# three discrete quality levels used by the rule-based / fixed baselines
LEVELS = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
LEVEL_RATE = [sent_rate(*l) for l in LEVELS]


class FixedPolicy:
    def __init__(self, level):
        self.a = encode(*LEVELS[level])
        self.name = ("fixed_low", "fixed_mid", "fixed_high")[level]

    def reset(self):
        pass

    def act(self, env):
        return self.a


class RulePolicy:
    """ABR-style threshold adapter: highest level whose rate fits the bandwidth
    estimate (margin m); step one level further down if smoothed RTT or the
    send-queue delay is above its threshold."""

    name = "rule"

    def __init__(self, rtt_hi=0.10, margin=0.85, qd_hi=0.08):
        self.rtt_hi, self.margin, self.qd_hi = rtt_hi, margin, qd_hi

    def reset(self):
        pass

    def act(self, env):
        r = env.raw()
        lvl = 0
        for l, rate in enumerate(LEVEL_RATE):
            if rate <= self.margin * r["bw"]:
                lvl = l
        if r["rtt"] > self.rtt_hi or r["qd"] > self.qd_hi:
            lvl = max(0, lvl - 1)
        return encode(*LEVELS[lvl])


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.ReLU(),
                               nn.Linear(128, N_ACT))

    def forward(self, x):
        return self.f(x)


class DQNPolicy:
    def __init__(self, qnet, name="dqn"):
        self.q = qnet
        self.name = name

    def reset(self):
        pass

    def act(self, env, state=None):
        s = env._state() if state is None else state
        with torch.no_grad():
            return int(self.q(torch.from_numpy(s).unsqueeze(0)).argmax(1).item())


def run_episode(policy, profile, seed, w_c=1.0, trace=False, cfg=None):
    env = TeleopEnv(profile, seed, w_c=w_c, cfg=cfg)
    policy.reset()
    done = False
    while not done:
        _, _, done = env.step(policy.act(env))
    m = env.metrics()
    if trace:
        return m, env
    return m


def train_dqn(seed, w_c, episodes=700, out_curve=None, cfg=None):
    """Double DQN, domain-randomised over the three training profiles."""
    torch.set_num_threads(1)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    q, tq = QNet(), QNet()
    tq.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = deque(maxlen=100_000)
    gamma, bs = 0.95, 128
    eps_decay = int(episodes * 0.45)
    steps, curve = 0, []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / eps_decay)
        prof = rng.choice(("stable", "congested", "highjitter"))
        env = TeleopEnv(prof, rng.randrange(10 ** 9), w_c=w_c, cfg=cfg)
        s = env._state()
        done = False
        while not done:
            if rng.random() < eps:
                a = rng.randrange(N_ACT)
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s).unsqueeze(0)).argmax(1).item())
            s2, r, done = env.step(a)
            buf.append((s, a, r, s2, float(done and not (env.n >= 1000))))
            s = s2
            steps += 1
            if len(buf) >= 1000 and steps % 2 == 0:
                b = random.sample(buf, bs)
                S = torch.from_numpy(np.stack([x[0] for x in b]))
                A = torch.tensor([x[1] for x in b])
                R = torch.tensor([x[2] for x in b], dtype=torch.float32)
                S2 = torch.from_numpy(np.stack([x[3] for x in b]))
                D = torch.tensor([x[4] for x in b], dtype=torch.float32)
                with torch.no_grad():
                    a2 = q(S2).argmax(1, keepdim=True)
                    tgt = R + gamma * (1 - D) * tq(S2).gather(1, a2).squeeze(1)
                pred = q(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, tgt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                if steps % 400 == 0:
                    tq.load_state_dict(q.state_dict())
        curve.append(env.ret)
    return q.state_dict(), curve
