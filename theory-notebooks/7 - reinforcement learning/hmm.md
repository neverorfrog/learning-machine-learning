# Hidden Markov Models

## Intuition

- It is a **Markov Chain** that evolves without forcing actions
  - By Markov Chain we mean a dynamic system where the future evolution depends only on the current state
  - Every state $x_t$ contains all information about the past
- Where the state is actually not directly observable
- So each state $x_t$ has a connected observation $z_t$

## Formal Definition

- Set of discrete states $x_t$ and (discrete or continuous) observations $z_t$
- Initial state is modeled under probability distribution $\pi_0=P(x_0)$
- Transition model: $A_{ij}=P(x_t=j|x_{t-1}=i)$
- Observation model: $b_k(z_t) = P(z_t|x_t=k)$
  - $z(t)$ can be a discrete value, or a gaussian

## Problems on HMM

- First of all, we can write the joint probability distribution between states and observations as $P(x=0:T,z=0:T)=P(x_0)P(z_0|x_0)P(x_1|x_0)P(z_1|x_1)...$
- Filtering
  - Estimate $P(x_k|z_{0:T})$, the current state given observations until now
- Smoothin
  - Estimate $P(x_T|z_{0:T})$, any past state given observations until now
- Solved by knowing the model's parameters
