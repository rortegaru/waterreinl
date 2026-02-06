#!/usr/bin/env python3
"""
RL Aquifer Prioritization V2.0 (from scratch)
- Gymnasium environment
- Tabular Q-learning (no neural networks)
- Discretized state space (bins)
- Simple, interpretable reward
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Utilities: discretization
# -----------------------------
def clip(x, lo, hi):
    return max(lo, min(hi, x))

def to_bin(x: float, edges: np.ndarray) -> int:
    """
    Convert continuous value x to a discrete bin index given edges.
    edges: array of bin edges of length (n_bins+1)
    """
    # np.digitize returns 1..n_bins; convert to 0..n_bins-1
    b = int(np.digitize([x], edges)[0]) - 1
    return int(clip(b, 0, len(edges) - 2))


@dataclass
class AquiferParams:
    # Ranges for continuous variables (used for normalization + clipping)
    V_min: float = 0.0     # Annual renewable volume (arbitrary units)
    V_max: float = 200.0
    A_min: float = -100.0  # Availability after extraction (can be negative)
    A_max: float = 100.0
    D_min: float = 0.0     # Distance (km)
    D_max: float = 200.0
    Q_min: float = 0.0     # Demand (L/s)
    Q_max: float = 500.0
    M_min: float = 0.0     # Modeling level (0..100)
    M_max: float = 100.0

    n_bins: int = 5

    # Dynamics step sizes (tune later)
    leak_repair_delta_Q: float = -30.0
    aqueduct_delta_D: float = -25.0
    aqueduct_delta_Q: float = +10.0  # operational burden
    dam_delta_A: float = +20.0
    study_delta_M: float = +20.0

    # Small stochasticity (kept minimal for reproducibility)
    noise_std: float = 2.0

    # Action costs (penalties)
    cost_leak: float = 0.5
    cost_aqueduct: float = 1.0
    cost_dam: float = 1.5
    cost_study: float = 0.8

    # Episode settings
    max_steps: int = 30

    # Terminal thresholds (define "acceptable state")
    A_target: float = 10.0
    D_target: float = 30.0
    Q_target: float = 120.0
    M_target: float = 70.0


class AquiferEnvV2(gym.Env):
    """
    Minimal RL environment:
    State: (V, A, D, Q, M) continuous internally, discretized externally.
    Actions:
      0 leak repair, 1 aqueduct, 2 dam, 3 study
    Reward: normalized linear utility - action cost
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, params: AquiferParams, seed: Optional[int] = 0):
        super().__init__()
        self.p = params
        self.rng = np.random.default_rng(seed)

        # Build bin edges for discretization
        self.edges_V = np.linspace(self.p.V_min, self.p.V_max, self.p.n_bins + 1)
        self.edges_A = np.linspace(self.p.A_min, self.p.A_max, self.p.n_bins + 1)
        self.edges_D = np.linspace(self.p.D_min, self.p.D_max, self.p.n_bins + 1)
        self.edges_Q = np.linspace(self.p.Q_min, self.p.Q_max, self.p.n_bins + 1)
        self.edges_M = np.linspace(self.p.M_min, self.p.M_max, self.p.n_bins + 1)

        # Discrete observation space: 5 vars each n_bins
        self.observation_space = spaces.MultiDiscrete([self.p.n_bins]*5)
        self.action_space = spaces.Discrete(4)

        self.state_cont = None
        self.steps = 0

    def _discretize(self, s: Tuple[float,float,float,float,float]) -> Tuple[int,int,int,int,int]:
        V, A, D, Q, M = s
        return (
            to_bin(V, self.edges_V),
            to_bin(A, self.edges_A),
            to_bin(D, self.edges_D),
            to_bin(Q, self.edges_Q),
            to_bin(M, self.edges_M),
        )

    def _normalize(self, x, lo, hi) -> float:
        if hi == lo:
            return 0.0
        return (x - lo) / (hi - lo)

    def _utility(self, V,A,D,Q,M) -> float:
        """
        Simple interpretable utility:
          + availability (higher is better)
          + modeling (higher is better)
          - distance (lower is better)
          - demand (lower is better)
        All normalized to [0,1] then combined.
        """
        A_n = self._normalize(clip(A, self.p.A_min, self.p.A_max), self.p.A_min, self.p.A_max)
        M_n = self._normalize(clip(M, self.p.M_min, self.p.M_max), self.p.M_min, self.p.M_max)
        D_n = self._normalize(clip(D, self.p.D_min, self.p.D_max), self.p.D_min, self.p.D_max)
        Q_n = self._normalize(clip(Q, self.p.Q_min, self.p.Q_max), self.p.Q_min, self.p.Q_max)

        # Availability + modeling good; distance + demand bad
        util = (+1.0*A_n + 0.8*M_n - 0.7*D_n - 0.7*Q_n)
        return util

    def _is_terminal(self, V,A,D,Q,M) -> bool:
        return (A >= self.p.A_target and
                D <= self.p.D_target and
                Q <= self.p.Q_target and
                M >= self.p.M_target)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0

        # Initialize a plausible stressed aquifer (you can later load from data)
        V = self.rng.uniform(30, 120)
        A = self.rng.uniform(-60, 10)
        D = self.rng.uniform(20, 160)
        Q = self.rng.uniform(80, 320)
        M = self.rng.uniform(0, 60)

        self.state_cont = (V,A,D,Q,M)
        obs = self._discretize(self.state_cont)
        info = {"state_cont": self.state_cont}
        return obs, info

    def step(self, action: int):
        self.steps += 1
        V,A,D,Q,M = self.state_cont

        # Apply deterministic deltas
        if action == 0:  # leak repair
            Q += self.p.leak_repair_delta_Q
            cost = self.p.cost_leak
        elif action == 1:  # aqueduct
            D += self.p.aqueduct_delta_D
            Q += self.p.aqueduct_delta_Q
            cost = self.p.cost_aqueduct
        elif action == 2:  # dam / augmentation
            A += self.p.dam_delta_A
            cost = self.p.cost_dam
        elif action == 3:  # study
            M += self.p.study_delta_M
            cost = self.p.cost_study
        else:
            raise ValueError("Invalid action")

        # Minimal noise
        if self.p.noise_std > 0:
            A += self.rng.normal(0, self.p.noise_std)
            D += self.rng.normal(0, self.p.noise_std)
            Q += self.rng.normal(0, self.p.noise_std)
            M += self.rng.normal(0, self.p.noise_std)

        # Clip to ranges
        V = clip(V, self.p.V_min, self.p.V_max)
        A = clip(A, self.p.A_min, self.p.A_max)
        D = clip(D, self.p.D_min, self.p.D_max)
        Q = clip(Q, self.p.Q_min, self.p.Q_max)
        M = clip(M, self.p.M_min, self.p.M_max)

        # Compute reward as change in utility (shaped) - cost
        util_before = self._utility(*self.state_cont)
        util_after = self._utility(V,A,D,Q,M)
        shaped = (util_after - util_before)

        reward = shaped - 0.05*cost

        terminated = self._is_terminal(V,A,D,Q,M)
        truncated = (self.steps >= self.p.max_steps)

        self.state_cont = (V,A,D,Q,M)
        obs = self._discretize(self.state_cont)
        info = {"state_cont": self.state_cont, "utility": util_after}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.state_cont is None:
            print("Env not reset.")
            return
        V,A,D,Q,M = self.state_cont
        print(f"Step={self.steps:02d} | V={V:.1f} A={A:.1f} D={D:.1f} Q={Q:.1f} M={M:.1f}")


# -----------------------------
# Q-learning (tabular)
# -----------------------------
def encode_state(obs: Tuple[int,int,int,int,int], n_bins: int) -> int:
    """
    Convert MultiDiscrete obs into single integer index.
    """
    v,a,d,q,m = obs
    idx = (((v*n_bins + a)*n_bins + d)*n_bins + q)*n_bins + m
    return idx

def train_q_learning(
    env: AquiferEnvV2,
    episodes: int = 5000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.999,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n_bins = env.p.n_bins
    n_states = n_bins**5
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    eps = eps_start
    returns = []

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        s = encode_state(obs, n_bins)

        done = False
        G = 0.0

        while not done:
            # epsilon-greedy
            if rng.random() < eps:
                a = int(rng.integers(0, n_actions))
            else:
                a = int(np.argmax(Q[s]))

            obs2, r, terminated, truncated, info2 = env.step(a)
            s2 = encode_state(obs2, n_bins)
            done = terminated or truncated

            # Q-learning update
            td_target = r + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s2
            G += r

        eps = max(eps_end, eps * eps_decay)
        returns.append(G)

    return Q, returns


def evaluate_policy(env: AquiferEnvV2, Q_table: np.ndarray, n_eval: int = 200, seed: int = 123):
    rng = np.random.default_rng(seed)
    n_bins = env.p.n_bins
    success = 0
    avg_return = 0.0
    avg_steps = 0.0

    for _ in range(n_eval):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        s = encode_state(obs, n_bins)
        done = False
        G = 0.0
        steps = 0

        while not done:
            a = int(np.argmax(Q_table[s]))
            obs2, r, terminated, truncated, info2 = env.step(a)
            s = encode_state(obs2, n_bins)
            done = terminated or truncated
            G += r
            steps += 1

        avg_return += G
        avg_steps += steps
        success += int(terminated)

    avg_return /= n_eval
    avg_steps /= n_eval
    success_rate = success / n_eval
    return {"success_rate": success_rate, "avg_return": avg_return, "avg_steps": avg_steps}


if __name__ == "__main__":
    params = AquiferParams(n_bins=5, max_steps=25, noise_std=1.5)
    env = AquiferEnvV2(params, seed=0)

    Q_table, returns = train_q_learning(env, episodes=6000)

    metrics = evaluate_policy(env, Q_table, n_eval=300)
    print("Evaluation:", metrics)
    print("Last 10 episode returns:", [round(x, 3) for x in returns[-10:]])

