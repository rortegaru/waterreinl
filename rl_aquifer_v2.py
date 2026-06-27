#!/usr/bin/env python3
"""
RL Aquifer Prioritization V2.0
- Gymnasium environment
- Tabular Q-learning (no neural networks)
- Discretized state space (bins)
- Runs on real CONAGUA/INEGI aquifer data
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# Utilities
# =============================================================================

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def to_bin(x: float, edges: np.ndarray) -> int:
    b = int(np.digitize([x], edges)[0]) - 1
    return int(clip(b, 0, len(edges) - 2))


# =============================================================================
# Data loading
# =============================================================================

def load_aquifer_data(filepath: str = "acuiferos_sdistance.xlsx") -> pd.DataFrame:
    """
    Load real CONAGUA aquifer data and map columns to RL variables.

    Variables returned:
      V  – annual recharge volume (R, hm³)
      A  – post-extraction availability (DMA, hm³; negative = overdrawn)
      D  – shortest distance to demand centre (km)
      Q  – water demand (Necesidad, hm³/year)
      M  – hydrogeological modelling level (0–100)
    """
    df = pd.read_excel(filepath)

    # Keep only overdrawn aquifers (DMA < 0)
    df = df[df["DMA"] < 0].copy()

    # Derive demand: 250 L/person/day × POBTOT × 365 days / 1e9  →  hm³/year
    df["Q_real"] = df["POBTOT"] * 250 * 365 / 1e9

    # Modeling level: no official column in main dataset → use 30 (moderate)
    df["M_real"] = 30.0

    out = df[["Acuifero", "Estado", "R", "DMA", "VEAS",
              "Shortest Distance", "POBTOT", "Q_real", "M_real"]].copy()
    out.columns = ["aquifer", "state", "V", "A", "VEAS",
                   "D", "POBTOT", "Q", "M"]
    return out.dropna(subset=["V", "A", "D", "Q"]).reset_index(drop=True)


# =============================================================================
# Environment parameters — calibrated to real data ranges
# =============================================================================

@dataclass
class AquiferParams:
    # --- Variable ranges (match real CONAGUA data) ---
    V_min: float = 0.0       # recharge volume hm³
    V_max: float = 600.0
    A_min: float = -700.0    # availability hm³ (negative = overdrawn)
    A_max: float = 50.0
    D_min: float = 0.0       # distance km
    D_max: float = 160.0
    Q_min: float = 0.0       # demand hm³/year
    Q_max: float = 60.0
    M_min: float = 0.0       # modelling level 0–100
    M_max: float = 100.0

    n_bins: int = 8           # more bins → finer Q-table

    # --- Action step sizes (meaningful at real-data scale) ---
    leak_repair_delta_Q:  float = -4.0    # reduce demand
    aqueduct_delta_D:     float = -20.0   # reduce distance
    aqueduct_delta_Q:     float = +1.5    # ops burden
    dam_delta_A:          float = +50.0   # increase availability (recharge/storage)
    study_delta_M:        float = +20.0   # improve hydrogeological knowledge

    noise_std: float = 1.0               # small stochastic noise

    # --- Action costs ---
    cost_leak:      float = 0.5
    cost_aqueduct:  float = 1.0
    cost_dam:       float = 1.5
    cost_study:     float = 0.8

    max_steps: int = 30

    # --- Terminal targets (define "acceptable management state") ---
    A_target: float = -10.0   # still slightly overdrawn but much improved
    D_target: float = 30.0
    Q_target: float = 5.0
    M_target: float = 70.0


# =============================================================================
# Gymnasium environment
# =============================================================================

class AquiferEnvV2(gym.Env):
    """
    State  : (V, A, D, Q, M) — continuous internally, binned for the agent.
    Actions: 0=leak repair, 1=aqueduct, 2=dam/recharge, 3=study
    Reward : shaped utility improvement minus action cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 params: AquiferParams,
                 aquifer_df: Optional[pd.DataFrame] = None,
                 seed: Optional[int] = 0):
        super().__init__()
        self.p = params
        self.aquifer_df = aquifer_df   # real data; None → random init
        self.rng = np.random.default_rng(seed)

        self.edges_V = np.linspace(self.p.V_min, self.p.V_max, self.p.n_bins + 1)
        self.edges_A = np.linspace(self.p.A_min, self.p.A_max, self.p.n_bins + 1)
        self.edges_D = np.linspace(self.p.D_min, self.p.D_max, self.p.n_bins + 1)
        self.edges_Q = np.linspace(self.p.Q_min, self.p.Q_max, self.p.n_bins + 1)
        self.edges_M = np.linspace(self.p.M_min, self.p.M_max, self.p.n_bins + 1)

        self.observation_space = spaces.MultiDiscrete([self.p.n_bins] * 5)
        self.action_space = spaces.Discrete(4)

        self.state_cont = None
        self.steps = 0
        self._current_aquifer_idx = None

    # ------------------------------------------------------------------
    def _discretize(self, s):
        V, A, D, Q, M = s
        return (
            to_bin(V, self.edges_V),
            to_bin(A, self.edges_A),
            to_bin(D, self.edges_D),
            to_bin(Q, self.edges_Q),
            to_bin(M, self.edges_M),
        )

    def _normalize(self, x, lo, hi):
        return (x - lo) / (hi - lo) if hi != lo else 0.0

    def _utility(self, V, A, D, Q, M) -> float:
        """
        Interpretable linear utility matching the paper's reward structure:
          + availability (higher → less overdrawn)
          + modelling    (higher → more knowledge)
          - distance     (lower → more accessible)
          - demand       (lower → less stressed)
        All normalized to [0, 1].
        """
        A_n = self._normalize(clip(A, self.p.A_min, self.p.A_max), self.p.A_min, self.p.A_max)
        M_n = self._normalize(clip(M, self.p.M_min, self.p.M_max), self.p.M_min, self.p.M_max)
        D_n = self._normalize(clip(D, self.p.D_min, self.p.D_max), self.p.D_min, self.p.D_max)
        Q_n = self._normalize(clip(Q, self.p.Q_min, self.p.Q_max), self.p.Q_min, self.p.Q_max)
        return 1.0 * A_n + 0.8 * M_n - 0.7 * D_n - 0.7 * Q_n

    def _reward(self, prev_state, next_state, cost: float) -> float:
        shaped = self._utility(*next_state) - self._utility(*prev_state)
        return shaped - 0.05 * cost

    def _is_terminal(self, V, A, D, Q, M) -> bool:
        return (A >= self.p.A_target and
                D <= self.p.D_target and
                Q <= self.p.Q_target and
                M >= self.p.M_target)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None, aquifer_idx: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0

        if self.aquifer_df is not None and aquifer_idx is not None:
            # Fixed aquifer from real data
            row = self.aquifer_df.iloc[aquifer_idx]
            V = float(clip(row["V"], self.p.V_min, self.p.V_max))
            A = float(clip(row["A"], self.p.A_min, self.p.A_max))
            D = float(clip(row["D"], self.p.D_min, self.p.D_max))
            Q = float(clip(row["Q"], self.p.Q_min, self.p.Q_max))
            M = float(clip(row["M"], self.p.M_min, self.p.M_max))
            self._current_aquifer_idx = aquifer_idx
        elif self.aquifer_df is not None:
            # Sample a random real aquifer
            idx = int(self.rng.integers(0, len(self.aquifer_df)))
            row = self.aquifer_df.iloc[idx]
            V = float(clip(row["V"], self.p.V_min, self.p.V_max))
            A = float(clip(row["A"], self.p.A_min, self.p.A_max))
            D = float(clip(row["D"], self.p.D_min, self.p.D_max))
            Q = float(clip(row["Q"], self.p.Q_min, self.p.Q_max))
            M = float(clip(row["M"], self.p.M_min, self.p.M_max))
            self._current_aquifer_idx = idx
        else:
            # Random initialization (for quick tests)
            V = float(self.rng.uniform(10, 300))
            A = float(self.rng.uniform(-300, -1))
            D = float(self.rng.uniform(5, 100))
            Q = float(self.rng.uniform(0.3, 20))
            M = float(self.rng.uniform(0, 50))
            self._current_aquifer_idx = None

        self.state_cont = (V, A, D, Q, M)
        return self._discretize(self.state_cont), {"state_cont": self.state_cont}

    def step(self, action: int):
        self.steps += 1
        V, A, D, Q, M = self.state_cont

        if action == 0:       # leak repair
            Q += self.p.leak_repair_delta_Q
            cost = self.p.cost_leak
        elif action == 1:     # aqueduct
            D += self.p.aqueduct_delta_D
            Q += self.p.aqueduct_delta_Q
            cost = self.p.cost_aqueduct
        elif action == 2:     # dam / recharge augmentation
            A += self.p.dam_delta_A
            cost = self.p.cost_dam
        elif action == 3:     # hydrogeological study
            M += self.p.study_delta_M
            cost = self.p.cost_study
        else:
            raise ValueError("Invalid action")

        if self.p.noise_std > 0:
            A += self.rng.normal(0, self.p.noise_std)
            D += self.rng.normal(0, self.p.noise_std)
            Q += self.rng.normal(0, self.p.noise_std)
            M += self.rng.normal(0, self.p.noise_std)

        V = clip(V, self.p.V_min, self.p.V_max)
        A = clip(A, self.p.A_min, self.p.A_max)
        D = clip(D, self.p.D_min, self.p.D_max)
        Q = clip(Q, self.p.Q_min, self.p.Q_max)
        M = clip(M, self.p.M_min, self.p.M_max)

        next_state = (V, A, D, Q, M)
        reward = self._reward(self.state_cont, next_state, cost)
        util_after = self._utility(*next_state)

        terminated = self._is_terminal(V, A, D, Q, M)
        truncated = (self.steps >= self.p.max_steps)

        self.state_cont = next_state
        obs = self._discretize(self.state_cont)
        return obs, reward, terminated, truncated, {"state_cont": next_state, "utility": util_after}

    def render(self):
        if self.state_cont is None:
            return
        V, A, D, Q, M = self.state_cont
        print(f"Step={self.steps:02d} | V={V:.1f} A={A:.1f} D={D:.1f} Q={Q:.2f} M={M:.1f}")


# =============================================================================
# Q-learning (tabular)
# =============================================================================

def encode_state(obs, n_bins: int) -> int:
    v, a, d, q, m = obs
    return (((v * n_bins + a) * n_bins + d) * n_bins + q) * n_bins + m


def train_q_learning(
    env: AquiferEnvV2,
    episodes: int = 8000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.9995,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n_bins = env.p.n_bins
    n_states = n_bins ** 5
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    eps = eps_start
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        s = encode_state(obs, n_bins)
        done = False
        G = 0.0

        while not done:
            a = int(rng.integers(0, n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = encode_state(obs2, n_bins)
            done = terminated or truncated
            td_target = r + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s2
            G += r

        eps = max(eps_end, eps * eps_decay)
        returns.append(G)

    return Q, returns


def evaluate_policy(env: AquiferEnvV2, Q_table: np.ndarray,
                    n_eval: int = 200, seed: int = 123):
    rng = np.random.default_rng(seed)
    n_bins = env.p.n_bins
    success, avg_return, avg_steps = 0, 0.0, 0.0

    for _ in range(n_eval):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        s = encode_state(obs, n_bins)
        done = False
        G, steps = 0.0, 0
        while not done:
            a = int(np.argmax(Q_table[s]))
            obs2, r, terminated, truncated, _ = env.step(a)
            s = encode_state(obs2, n_bins)
            done = terminated or truncated
            G += r
            steps += 1
        avg_return += G
        avg_steps += steps
        success += int(terminated)

    n = n_eval
    return {
        "success_rate": success / n,
        "avg_return":   avg_return / n,
        "avg_steps":    avg_steps / n,
    }


# =============================================================================
# Real-data prioritization
# =============================================================================

def prioritize_aquifers(env: AquiferEnvV2,
                        Q_table: np.ndarray,
                        aquifer_df: pd.DataFrame,
                        n_runs: int = 50) -> pd.DataFrame:
    """
    Evaluate every real aquifer with the trained greedy policy.
    Returns a DataFrame ranked by cumulative reward (benefit score).
    """
    n_bins = env.p.n_bins
    records = []

    for idx in range(len(aquifer_df)):
        returns = []
        for run in range(n_runs):
            obs, _ = env.reset(seed=run * 1000 + idx, aquifer_idx=idx)
            s = encode_state(obs, n_bins)
            done = False
            G, steps = 0.0, 0
            while not done:
                a = int(np.argmax(Q_table[s]))
                obs2, r, terminated, truncated, _ = env.step(a)
                s = encode_state(obs2, n_bins)
                done = terminated or truncated
                G += r
                steps += 1
            returns.append(G)

        row = aquifer_df.iloc[idx]
        records.append({
            "aquifer":    row["aquifer"],
            "state":      row["state"],
            "V_hm3":      round(row["V"], 2),
            "A_hm3":      round(row["A"], 2),
            "D_km":       round(row["D"], 2),
            "Q_hm3":      round(row["Q"], 3),
            "M_level":    int(row["M"]),
            "benefit":    round(float(np.mean(returns)), 4),
            "benefit_std": round(float(np.std(returns)), 4),
        })

    result = pd.DataFrame(records)
    result = result.sort_values("benefit", ascending=False).reset_index(drop=True)
    result.index += 1
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RL Aquifer Prioritization — Real Data Run")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1] Loading aquifer data...")
    try:
        df = load_aquifer_data("acuiferos_sdistance.xlsx")
    except FileNotFoundError:
        df = load_aquifer_data("ACUIFEROS_TABLAS/acuiferos_sdistance.xlsx")

    print(f"    Overdrawn aquifers loaded: {len(df)}")
    print(f"    States: {df['state'].nunique()}")
    print(f"    Variable ranges:")
    for col, label in [("V","V (recharge hm³)"), ("A","A (DMA hm³)"),
                       ("D","D (dist km)"), ("Q","Q (demand hm³)")]:
        print(f"      {label:25s}  {df[col].min():.2f} – {df[col].max():.2f}")

    # ── 2. Set up environment with real-data ranges ───────────────
    params = AquiferParams(
        V_min=0.0,    V_max=600.0,
        A_min=-700.0, A_max=50.0,
        D_min=0.0,    D_max=160.0,
        Q_min=0.0,    Q_max=60.0,
        M_min=0.0,    M_max=100.0,
        n_bins=8,
        max_steps=30,
        noise_std=0.5,
        dam_delta_A=50.0,
        study_delta_M=20.0,
        leak_repair_delta_Q=-4.0,
        aqueduct_delta_D=-20.0,
    )
    env = AquiferEnvV2(params, aquifer_df=df, seed=42)

    # ── 3. Train Q-table ──────────────────────────────────────────
    print("\n[2] Training Q-learning policy on real aquifer data...")
    Q_table, returns = train_q_learning(env, episodes=10000, seed=42)
    print(f"    Training complete.")
    print(f"    Last 10 episode returns: {[round(x,3) for x in returns[-10:]]}")

    # ── 4. Evaluate trained policy ────────────────────────────────
    print("\n[3] Evaluating trained policy...")
    metrics = evaluate_policy(env, Q_table, n_eval=400)
    print(f"    Success rate : {metrics['success_rate']:.1%}")
    print(f"    Avg return   : {metrics['avg_return']:.4f}")
    print(f"    Avg steps    : {metrics['avg_steps']:.1f}")

    # ── 5. Prioritize all real aquifers ───────────────────────────
    print("\n[4] Scoring all overdrawn aquifers (this may take ~30s)...")
    ranking = prioritize_aquifers(env, Q_table, df, n_runs=30)

    print("\n=== TOP 20 PRIORITY AQUIFERS ===")
    print(ranking[["aquifer","state","A_hm3","D_km","Q_hm3","benefit"]].head(20).to_string())

    # ── 6. Summary by state ───────────────────────────────────────
    print("\n=== PRIORITY BY STATE (mean benefit score) ===")
    state_rank = (ranking.groupby("state")["benefit"]
                  .agg(["mean","count","max"])
                  .sort_values("mean", ascending=False)
                  .round(4))
    state_rank.columns = ["mean_benefit", "n_aquifers", "top_benefit"]
    print(state_rank.head(20).to_string())

    # ── 7. Save results ───────────────────────────────────────────
    out_path = "aquifer_priority_ranking.csv"
    ranking.to_csv(out_path, index=True)
    print(f"\n[5] Full ranking saved to: {out_path}")
    print("=" * 60)
