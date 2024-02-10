# Parallel learning

- Models are trained in parallel and their predictions are combined

## Voting schemes

- The combination consists in taking a weighted average between outputs of different models
- They differ basically by how the voting weights are chosen
- Regression
  - Simply a weighted average $y(x)=\Sigma w_m y_m(x)$
- Classification
  - We count how many times each different class has been "voted" by weighting the vote of each model and then take the maximum: $y(x)=\argmax_c \Sigma_m w_m (y_m(x)==c)$

### Vanilla Voting

- Fixed weights
- Constraint: $\Sigma w_m=1$

### Mixture of experts

- Voting weights change depending on the input through a non-linear gating function
- Constraint: $\Sigma w_m=1$

### Stacking

- The voting function itself can be learned

## Cascading scheme

- Different approach than weighted average, where the final output does not correspond to any of the single outputs
- Here, instead, we have some treshold on each model's output, and the first the satisfies this threshold makes it through as the final output of the multiple learner
- Order here is also relevant

## Bagging

- Wants to avoid overfitting
- Procedure
  - Instead of training each model on the entire dataset, we generate a set of $M$ subsets of the dataset called bootstrap dataset $D_m$ (not a partition)
  - We then train a model on each bootstrap dataset $D_m$
  - We then take a vote among the predictions through any voting scheme
- How are the bootstrap dataset generated?
  - Usually with random sampling with replacement
- How does this reduce overfitting?
  - None of the models sees the entire dataset
  - The generated solution is more general
- Example?
  - Random forests, where for each $D_m$ we generate a bunch of decision trees and combine their results
