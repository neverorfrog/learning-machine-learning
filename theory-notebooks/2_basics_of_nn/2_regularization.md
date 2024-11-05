# Practical Issues


## Bias-Variance (Over-Underfitting)
- Bias: we make assumptions about the data
  - High Bias $\rightarrow$ Low representational power
  - Underfitting if to high
- Variance: the exact opposite
  - Overfitting if too high

## Double Descent
- High-dimensional data is very sparse
- Difficult to infer relations between datapoints
- As a consequence, the model actually performs better when overparametrizing (does not overfit)
- But there is a critical regime, where test error is the highest
  - Where capacity is just as high to match data intricacies

## How do we mitigate these problems? REGULARIZATION

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


### Explicit Regularization
- Adds a term proportional to the weights to the cost function
- What is the effect related to overfitting?
  - Variance is decreased
  - Bias is increased
  - Function space becomes smaller

### Early Stopping
- Monitor the loss on the validation set
- Stop if it starts increasing

### Data Augmentation
- Adding noise to data through transofrmation
