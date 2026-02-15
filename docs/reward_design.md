# Reward Design

The reward is defined as:

reward = U(next_state) − U(current_state) − cost

Where utility:

U = +A +0.8M −0.7D −0.7Q

Important interpretation:

High reward does NOT mean good aquifer.

High reward means large possible improvement.

Therefore:
critical aquifers generate higher rewards
stable aquifers generate near-zero rewards

The RL agent learns improvement gradient, not final condition.
