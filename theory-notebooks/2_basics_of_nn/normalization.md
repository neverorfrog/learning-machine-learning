# Normalization

## Batch Normalization 

- Why?
  - Standardized data (zero mean, unit variance) is more convenient for optimizers
    - Based on standard practice to normalize data by doing $x'=\frac{x-\mu}{\sqrt\sigma^2}$
  - To avoid big weight changes from one layer to another
  - To introduce some sort of noise into the network
- How?
  - Basic steps
    - We estimate mean and variance of data based only on the current minibatch
    - Then we apply the formula above to individual layers by making it a layer itself
    - Then we also apply scale and shift.
  - In **test mode**
    - Mean and variance are estimated using the whole dataset
    - Becomes a linear operator
  - In **convolutional layers** the mean and variance are computed for each channel
- Consequences
  - Noise injection (estimating mean and variance is noisy) acts as a sort of regularization
  - Empirically found that it allows for higher learning rates and better convergence
