# Learning in the presence of Noisy Labels

- Classification setting
- We are interested in label noise

## How do we model label noise?
- By a noise transition matrix $T$
- Element $T_{ij}$ indicates the probability that true label $i$ is flipped into $j$
### Symmetric
- Label is flipped to every other label with same probability
### Antisymmetric
- Label is flipped to one particular label with more probability

## How do we mitigate the problem?
### Robust Loss Functions
- MAE
  - Treats all errors equally
- generalized cross-entropy (expansion of MAE)
  - Introduces a hyperparameter q that influences how much the loss listens to noise
### Regularization
- Mix-up
  - Do a convex combination between two samples
### Early Learning Regularization
- NN tend to learn easier pattern first (so where there is a clean sample)
- Correct labels vanish while the training goes on
- Use early learning model to neutralize effect of incorrect labels on the gradient
### Co-Teaching
- Error flows can be reduced by peer networks mutually
### Outlier Discounting

### Multiple Annotations


