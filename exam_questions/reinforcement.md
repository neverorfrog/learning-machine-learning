# Reinforcement Learning

## Difference with supervised

- Define the difference between supervised and reinforcement
  - In supervised learning we are given a dataset $D=\{x_n,t_n\}$, where $t_n$ are labels and the problem is to learn a function $f: X \rightarrow T$ in form of a parametrized function $y(\theta)$. This is typically done with the help of an error function $J(\theta)$ that gives a measure of similarity between what comes from the learnt function and what should be the actual output.
  - RL has no labels in the dataset, which is instead a temporal series of states, actions and rewards like $\{s_0,a_0,r_1,...\}$. In this context, the problem of RL is to map an action to each state through the policy $\pi(s)$ with the goal to maximise the (expected) discounted return, defined as $G_t = r_t + \lambda r_{t+1} + ...$
  - So, the difference lies in the dataset structure, but also in the different performance measure
- Describe the full-observability property in MDP and its relation to non-deterministic outcomes of actions
  - In an MDP, we say to have full observability if we can directly access the state in its entirety and without noise, which means we don't need any observation model $o(s'|s,a)$
  - If we have a non-deterministic MDP, what changes is that we have to wait for the effect of an action before we can actually observe its outcome. Thus, non-deterministic actions don't hinder observability, but create a sort of "delay" in the MDP. The reward model becomes $r(s',s,a)$ instead of just $r(s,a)$