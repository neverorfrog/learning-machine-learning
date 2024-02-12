# Partially Observable Markov Decision Process

## Intuition

- Basically HMM with actions
- Or conversely MDP with observation
- Ingredients:
  - Set of states $X$
  - Set of actions $A$
  - Transition probability distribution $P(x'|x,a)$
  - Observation probability distribution $P(z'|x,a)$
  - Reward function $\rho(x',x,a)$
  - Initial state distribution $P(x_0)$

## First Solution Concept: Belief States

- We **cannot use states directly**, which means we cannot directly associate an action to a state to build up a policy
- Thus, we need to introduce **belief states** $b(x)$ as probability distributions over the states
  - $b(x)$ embeds the observation
- So the ingredients become:
  - Set of belief states $B$
  - Set of actions $A$
  - Transition probability distribution $P(b'|b,a)$
  - Reward function $\rho(b',b,a)$
  - Initial state distribution $P(b_0)$
- **Problem**: the belief state space is exponential wrt the plain state space
  - So we need to discretize them

## Second Solution Concept: Policy Tree

- A tree where nodes are actions and edges
- We construct it and get the optimal policy by taking the maximum expected reward while exploring different policy trees
- The difference to normal MDPs is that one action could be repeated multiple times, in order to increase the "belief" of the agent that one observation corresponds to a real state
