# Building Deeper Neural Networks

## Problem

- Deep neural networks have long chains of derivatives
- **Shattered gradient**
    - When doing backpropagation, the gradient can change drastically by traversing the whole network
    - Slows learning down
- **Vanishing gradient**
  - The longer the chain the bigger the effect of a tiny change over the whole chain
  - Slows learning down
- **Exploding gradient**
  - Variance doubles at every layer
- **Overfitting**
  - Deeper models comprise a richer function space
  - Thus, there is a risk that test error increases
- **Optimization**
  - Optimizers have difficulties when the function space becomes very big

## Residual Connections (or how to avoid shattered gradients)

- What?
  - $h(x)=f(x)+x$ where $f(x)$ is the function applied by the convolution block
  - So, the input contributes directly to the output
  - Intuitively, the layer learns a thing (the residual) to add to the input, in order to obtain the output
- Why?
  - To have connections through which the gradient flows unchanged
    - If the input contributes directly to the gradient, no shattering f the gradient
  - Tending towards nested function space
    - residual block before relu learns $f(x)=h(x)-x$
    - each layer tends towards the identity function instead of enforcing a transformation
    - thus, the residual block wants to learn 0 if $h(x)$ has to be the identity, and that is very easy for optimization
    - the layer learns the residual mapping $f(x)$ and not the layer's output $h(x)$
- What is done in practice?
  - To have the same number of channels as the input, we add a 1x1 convolutional layer to the residual connection
    - This is also called **bottleneck**
- Intuitions:
  - Backpropagation
    - In the forward pass, the input passes through the skip connection unmodified
    - In the backward pass, the gradient is simply summed to the previously computed gradient
  - Composition
    - The output of the model is a sum of the outputs of many submodels

## Batch Normalization (or how to avoid exploding gradients)

- Why?
  - Even with residual connections, gradients may explode
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

### Consequences
  - Noise injection (estimating mean and variance is noisy) acts as a sort of regularization
  - Empirically found that it allows for higher learning rates and better convergence