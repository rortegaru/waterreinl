# RL Aquifer Prioritization

Reinforcement Learning environment for prioritizing groundwater management interventions using a transparent tabular Q-learning model.

This project does NOT attempt to simulate hydrology.  
It evaluates management decisions under a simplified decision utility framework.

---

## Concept

Each aquifer is described by 5 variables:

- V: Renewable volume
- A: Availability after extraction
- D: Distance to demand center
- Q: Water demand
- M: Hydrogeological knowledge level

The agent applies actions (interventions):

0 - Leak repair  
1 - Aqueduct  
2 - Dam / augmentation  
3 - Hydrogeological study  

The goal is to reach an acceptable management state.

---

## Reward definition

The reward is NOT absolute performance.

It is improvement:

reward = Utility(next_state) − Utility(current_state) − action_cost

Therefore:

- Large rewards → critical systems (large improvement possible)
- Small rewards → stable systems

The model prioritizes urgency, not quality.

---

## Interpretation

Final cumulative reward is used as a stress indicator:

Lower return → higher priority aquifer

The RL agent does not decide infrastructure policy.  
It reveals which intervention reduces system stress faster under the assumed utility weights.

---

## Running

``bash
python rl_aquifer_v2_1.py


## Requirements

Install dependencies:

pip install -r requirements.txt

## Documentation

Detailed technical explanation of the model:

- Methodology → [docs/methodology.md](docs/methodology.md)
- Environment meaning → [docs/environment.md](docs/environment.md)
- Reward design → [docs/reward_design.md](docs/reward_design.md)
- Result interpretation → [docs/interpretation.md](docs/interpretation.md)


Important scientific note

The results depend strongly on the utility function weights:

U = +A +0.8M −0.7D −0.7Q

Changing these weights changes the recommended interventions.
This repository should be interpreted as a decision-analysis framework, not a hydrological simulator.
