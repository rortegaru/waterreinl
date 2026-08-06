# Methodology

## Problem formulation

The system is modeled as a Markov Decision Process (MDP):

State S = (V, A, D, Q, M)

Where:
- V: Renewable groundwater volume
- A: Availability after extraction
- D: Distance to demand center
- Q: Water demand
- M: Hydrogeological knowledge level

Actions represent management interventions:

0: Leak repair  
1: Aqueduct construction  
2: Dam / recharge augmentation  
3: Hydrogeological study

The objective is not hydrological simulation.  
The objective is decision prioritization.

---

## Why tabular Q-learning

The state space is discretized into bins to obtain:

finite states = n_bins^5

This allows:
- full transparency
- interpretability
- no black-box behavior
- reproducible policies

The agent learns a policy that maximizes cumulative improvement.
