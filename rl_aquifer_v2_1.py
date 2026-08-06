#!/usr/bin/env python3
"""
RL Aquifer Prioritization V2.0 _ 1 (from scratch)
- Gymnasium environment
- Tabular Q-learning (no neural networks)
- Discretized state space (bins)
- Simple, interpretable reward
"""
# =============================================================================
# RL Aquifer Prioritization V2.0 — How the environment works (detailed guide)
# =============================================================================
#
# OVERVIEW
# --------
# This file defines a minimal Gymnasium environment + tabular Q-learning.
# The environment is intentionally simple and transparent:
#
#   - The environment stores an internal continuous state (V, A, D, Q, M).
#   - The agent observes a discretized (binned) version of the state.
#   - The agent chooses one of 4 actions (interventions).
#   - Each action changes one or two variables (A, D, Q, M).
#   - The environment returns:
#       obs: discretized state (MultiDiscrete bins)
#       reward: a scalar computed inside step()
#       terminated: True if targets are met (acceptable management state)
#       truncated: True if max_steps reached
#       info: continuous state + current utility
#
#
# STATE VARIABLES (continuous, internal)
# -------------------------------------
# V = Annual Volume (renewable groundwater volume)
# A = Availability (remaining volume after extraction; can be negative)
# D = Distance (km) between resource and demand center
# Q = Demand (L/s)
# M = Modeling (hydrogeological understanding level, 0..100)
#
# NOTE:
# - In this V2.0 draft, V is kept constant within an episode (not changed by actions).
# - A, D, Q, M are changed by actions + small noise.
#
#
# OBSERVATION (what the agent sees)
# --------------------------------
# The observation is NOT the continuous state.
# It is a discretized (binned) version:
#
#   obs = (v_bin, a_bin, d_bin, q_bin, m_bin)
#
# Each *_bin is an integer in [0, n_bins-1], produced by to_bin(x, edges).
#
# Discretization edges are built with np.linspace(min, max, n_bins+1).
#
#
# ACTIONS (Discrete(4))
# --------------------
# Actions represent simple interventions:
#
#   0: leak repair            -> reduces demand Q (good), small cost
#   1: aqueduct               -> reduces distance D (good) but increases Q slightly (ops burden), cost
#   2: dam / augmentation     -> increases availability A (good), higher cost
#   3: hydrogeological study  -> increases modeling M (good), cost
#
# Each action applies a fixed delta (step size), plus optional Gaussian noise.
#
#
# REWARD (IMPORTANT: where it is computed)
# ---------------------------------------
# There is NO separate reward() function.
# The reward is computed inside AquiferEnvV2.step(action).
#
# The environment defines an internal scalar "utility" function:
#
#   utility = + wA * A_normalized
#             + wM * M_normalized
#             - wD * D_normalized
#             - wQ * Q_normalized
#
# where each variable is normalized into [0,1] using its min/max ranges.
#
# In step(), the reward is NOT the absolute utility.
# Instead it is a *shaped reward* based on utility improvement:
#
#   util_before = utility(current_state)
#   util_after  = utility(next_state)
#   shaped      = util_after - util_before
#
# Then we subtract a small action cost:
#
#   reward = shaped - 0.05 * action_cost
#
# So: reward is positive if the action improves utility enough to overcome the cost.
#
#
# TERMINATION (done conditions)
# -----------------------------
# terminated = True if the state meets all targets:
#
#   A >= A_target
#   D <= D_target
#   Q <= Q_target
#   M >= M_target
#
# truncated = True if steps >= max_steps
#
#
# WHY SHAPED REWARD?
# ------------------
# Using reward = (util_after - util_before) makes learning easier because:
# - the agent gets immediate feedback for improvement
# - it does not need to wait until the end of the episode to know it is doing well
# - it encourages monotonic improvement toward targets
#
# If you want "ranking of scenarios" as a final score, you can later switch to:
# - absolute utility as reward, or
# - cumulative absolute utility, or
# - cumulative shaped reward but evaluated under a fixed policy across scenarios
# =============================================================================


from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

# 1. Cargar el dataset antes de iniciar el entorno
df_acuiferos = pd.read_excel('acuiferos_DMA_benefits.xlsx')


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
# =============================================================================
# AquiferEnvV2 — step-by-step mechanics (complete guide)
# =============================================================================
#
# PURPOSE
# -------
# This environment is a deliberately minimal, transparent RL testbed designed to:
#   (i) discretize a multi-variable aquifer management state,
#  (ii) apply a small set of interpretable interventions (actions),
# (iii) compute a scalar reward that supports tabular Q-learning,
#  (iv) terminate when management targets are reached or time runs out.
#
# INTERNAL STATE (continuous)
# ---------------------------
# The environment keeps a continuous state internally:
#   V = Annual renewable volume (typically fixed per aquifer; from CONAGUA)
#   A = Availability after extraction (can be negative)
#   D = Distance from resource to demand center (km)
#   Q = Demand (L/s)
#   M = Modeling level / hydrogeological understanding (0..100)
#
# In the current code, V is sampled randomly in reset(); later it should be fixed
# per aquifer (scenario parameter). A, D, Q, M evolve through actions + noise.
#
# OBSERVATION (discretized / binned)
# ---------------------------------
# The agent DOES NOT observe continuous values. Instead, each variable is mapped
# into a discrete bin index in [0, n_bins-1] using fixed bin edges:
#   edges_X = linspace(X_min, X_max, n_bins+1)
#   x_bin = digitize(x, edges_X) - 1
#
# Observation returned by reset()/step() is:
#   obs = (v_bin, a_bin, d_bin, q_bin, m_bin)
# and observation_space = MultiDiscrete([n_bins]*5)
#
# ACTION SPACE (4 interventions)
# ------------------------------
# action 0: leak repair
#   - decreases Q by leak_repair_delta_Q
#   - incurs cost_leak
#
# action 1: aqueduct
#   - decreases D by aqueduct_delta_D
#   - increases Q slightly by aqueduct_delta_Q (operational burden)
#   - incurs cost_aqueduct
#
# action 2: dam / augmentation
#   - increases A by dam_delta_A
#   - incurs cost_dam
#
# action 3: hydrogeological study
#   - increases M by study_delta_M
#   - incurs cost_study
#
# STATE UPDATE ORDER IN step()
# ----------------------------
# step(action) executes the following sequence:
#
# (1) Read current continuous state: (V, A, D, Q, M)
# (2) Apply deterministic action deltas to A/D/Q/M (V unchanged)
# (3) Add small Gaussian noise to A/D/Q/M (optional, noise_std)
# (4) Clip all variables into predefined ranges [min, max]
# (5) Compute reward from utility change ("shaped reward") minus action cost
# (6) Check termination criteria (targets reached) and truncation (max_steps)
# (7) Save next continuous state
# (8) Return discretized observation, reward, terminated, truncated, info
#
# UTILITY FUNCTION (scalar, normalized)
# -------------------------------------
# A helper scalar "utility" is computed from normalized variables:
#   A_n = normalize(A) in [0,1]
#   M_n = normalize(M) in [0,1]
#   D_n = normalize(D) in [0,1]
#   Q_n = normalize(Q) in [0,1]
#
# Utility is a linear combination:
#   utility = +1.0*A_n + 0.8*M_n - 0.7*D_n - 0.7*Q_n
#
# REWARD (where it actually is)
# -----------------------------
# IMPORTANT: there is no standalone reward() function in the current code.
# The reward is computed inside step() using "reward shaping":
#
#   util_before = utility(current_state)
#   util_after  = utility(next_state)
#   shaped      = util_after - util_before
#
# Then a small action penalty is applied:
#   reward = shaped - 0.05*cost
#
# Therefore reward is positive if the action improves utility enough to offset cost.
#
# TERMINATION AND TRUNCATION
# --------------------------
# terminated = True if ALL targets are met simultaneously:
#   A >= A_target, D <= D_target, Q <= Q_target, M >= M_target
#
# truncated = True if steps >= max_steps
#
# TABULAR Q-LEARNING PIPELINE
# ---------------------------
# - obs (MultiDiscrete) is encoded into a single integer index via encode_state()
# - Q-table shape: (n_bins^5, n_actions)
# - epsilon-greedy exploration selects actions during training
# - Q-learning update:
#     Q[s,a] <- Q[s,a] + alpha * (r + gamma*max_a' Q[s',a'] - Q[s,a])
#
# EVALUATION
# ----------
# evaluate_policy() runs greedy actions (argmax Q[s]) over many episodes and reports:
#   - success_rate: fraction of terminated episodes (targets reached)
#   - avg_return: average cumulative reward
#   - avg_steps: average number of steps per episode
# =============================================================================


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
        Decision utility (not hydrological performance).

        This function encodes management preference assumptions.
        It does not represent groundwater physics.

        Higher values indicate better management condition,
        not higher water level.

        Changing weights changes recommended infrastructure strategy.
        """

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

    def _reward(self, prev_state, next_state, cost: float) -> float:
        """
        Scientific meaning:

        The reward measures marginal improvement between states.
        Large reward indicates a system far from equilibrium.

        Therefore high reward corresponds to critical aquifers,
        not healthy ones.

        The agent learns improvement gradient, not final quality.
        """

        util_before = self._utility(*prev_state)
        util_after  = self._utility(*next_state)
        shaped = util_after - util_before
        return shaped - 0.05 * cost


    def _is_terminal(self, V,A,D,Q,M) -> bool:
        return (A >= self.p.A_target and
                D <= self.p.D_target and
                Q <= self.p.Q_target and
                M >= self.p.M_target)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0

        # 2. SELECCIÓN DE DATOS REALES
        # En la función reset, cambia estas líneas:
        idx = self.rng.integers(0, len(df_acuiferos))
        row = df_acuiferos.iloc[idx]

        # MAPEADO CORRECTO:
        V = float(row['VEAS'])                      # Antes era 'V'
        A = float(row['DMA'])                       # Antes era 'A'
        D = float(row['Shortest Distance'])         # Antes era 'D'
        Q = float(row['Necesidad (pobtot*lt)hm3'])  # Antes era 'Q'
        
        # Como 'M' (Modelado) no está en tu Excel, 
        # podemos inventar un valor inicial o ponerlo en 0:
        M = float(self.rng.uniform(10, 30))

        self.state_cont = (V, A, D, Q, M)
        obs = self._discretize(self.state_cont)
        info = {"state_cont": self.state_cont, "acuifero_name": row.get('nombre', idx)}
        return obs, info
    
    def reset_specific_aquifer(self, index: int):
        """Fuerza el inicio del ambiente en un acuífero específico del dataframe"""
        self.steps = 0
        row = df_acuiferos.iloc[index]
        
        V = float(row['VEAS'])
        A = float(row['DMA'])
        D = float(row['Shortest Distance'])
        Q = float(row['Necesidad (pobtot*lt)hm3']) * 31.71 # Conversión a L/s
        M = 15.0 # Conocimiento base igual para todos
        
        self.state_cont = (V, A, D, Q, M)
        return self._discretize(self.state_cont), {"state_cont": self.state_cont}
    
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

        next_state = (V, A, D, Q, M)
        reward = self._reward(self.state_cont, next_state, cost)
        util_after = self._utility(*next_state)  # keep for info/debug

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


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Configuración e Inicialización
    params = AquiferParams(n_bins=5, max_steps=25, noise_std=1.5)
    env = AquiferEnvV2(params, seed=42)
    
    # 2. ENTRENAMIENTO (Curva de aprendizaje)
    print(f"Entrenando agente en escenarios aleatorios...")
    Q_table, returns_history = train_q_learning(env, episodes=8000)
    
    # --- EVALUACIÓN Y RECOLECCIÓN DE DATOS ---
    results = []
    print(f"Analizando sensibilidad y clusters para {len(df_acuiferos)} acuíferos...")

    for i in range(len(df_acuiferos)):
        obs, info = env.reset_specific_aquifer(i)
        s = encode_state(obs, params.n_bins)
        
        total_reward = 0
        actions_taken = []
        done = False
        
        while not done:
            a = int(np.argmax(Q_table[s]))
            actions_taken.append(a)
            obs, r, term, trunc, info = env.step(a)
            s = encode_state(obs, params.n_bins)
            total_reward += r
            done = term or trunc
        
        results.append({
            "Acuifero": df_acuiferos.iloc[i]['Acuifero'],
            "Reward_Final": total_reward,
            "Accion_Principal": max(set(actions_taken), key=actions_taken.count) if actions_taken else -1
        })

    df_res = pd.DataFrame(results).sort_values(by="Reward_Final")

    # --- FIGURA 1: CURVA DE APRENDIZAJE ---
    plt.figure(figsize=(10, 5))
    plt.plot(pd.Series(returns_history).rolling(window=100).mean(), color='blue')
    plt.title("Progreso del Entrenamiento (Curva de Aprendizaje)")
    plt.xlabel("Episodios")
    plt.ylabel("Reward Promedio (Suavizado)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- FIGURA 2: COMPARATIVA TOP 10 (CRÍTICOS VS SANOS) ---
    top_criticos = df_res.head(10)
    top_sanos = df_res.tail(10)
    df_comparativa = pd.concat([top_criticos, top_sanos])

    plt.figure(figsize=(12, 6))
    colores_barras = ['red']*10 + ['green']*10
    plt.barh(df_comparativa['Acuifero'], df_comparativa['Reward_Final'], color=colores_barras)
    plt.axvline(x=0.40, color='black', linestyle='--', alpha=0.5)
    plt.title("Acuíferos Prioritarios: Los 10 más Críticos vs 10 más Estables")
    plt.xlabel("Reward Acumulado (Menor = Más Urgente)")
    plt.tight_layout()
    plt.show()

    # --- FIGURA 3: HISTOGRAMA DE PRIORIDADES (CLUSTERS) ---
    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(df_res['Reward_Final'], bins=30, edgecolor='black', alpha=0.7)

    for patch in patches:
        if patch.get_x() < 0.40: patch.set_facecolor('red')
        elif patch.get_x() < 0.42: patch.set_facecolor('yellow')
        else: patch.set_facecolor('green')

    plt.title("Distribución General: Segmentación por Niveles de Estrés Hídrico")
    plt.xlabel("Reward Final")
    plt.ylabel("Frecuencia (Acuíferos)")
    plt.show()

    # --- FIGURA 4: SENSIBILIDAD (POLÍTICA DE LA IA) ---
    nombres_acciones = {0: "Reparar Fugas", 1: "Acueducto", 2: "Presa/Mejora", 3: "Estudio Hidro."}
    estrategia_criticos = df_res.head(20)['Accion_Principal'].map(nombres_acciones).value_counts()

    plt.figure(figsize=(8, 5))
    estrategia_criticos.plot(kind='pie', autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("¿Cuál es la solución que propone la IA para los acuíferos más críticos?")
    plt.ylabel("")
    plt.show()

    # --- EXPORTAR RESULTADOS ---
    df_res.to_excel("priorizacion_completa_acuiferos.xlsx", index=False)
    print("Archivo 'priorizacion_completa_acuiferos.xlsx' generado con éxito.")
# plt.show()  <-- Comenta esta línea

