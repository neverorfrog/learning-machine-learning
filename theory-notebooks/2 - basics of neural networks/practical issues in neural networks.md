# Practical Issues

## Overfitting

### Dropout

- What? Dropout randomly clamps a subset (typically 50%) of hidden units to zero at each iteration of SGD
- Why? Network is less dependent on any given hidden unit and smaller weights so that their variation influences less strongly the function
- How?
  - **Training**:
    - **Bernoulli**: $\tilde H=M \odot H$
      - $M$ is boolean matrix from Bernoulli distribution
      - $H$ is output from layer
    - **Inverted Dropout**: $\tilde H=\frac{H \odot M}{1-p}$
  - **Inference**:
    - **Montecarlo (for Bernoulli)**: run N inferences and take the expected value as final output
      - It's as if we would test N different models
    - **Nothing (for inverted)**: we just infer as if dropout wouldn't be there
- When?
  - Early dropout counteracts high variations in SGD
  - Late dropout can be beneficial for overfitting

### Regularization

- TODO

### Early Stopping

- TODO
