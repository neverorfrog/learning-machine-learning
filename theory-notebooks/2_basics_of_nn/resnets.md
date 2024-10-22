# ResNet

## Problem

- Deep neural networks have long chains of derivatives
- **Shattered gradient**
    - When doing backpropagation, the gradient can change drastically by traversing the whole network
    - Slows learning down
- **Vanishing gradient**
  - The longer the chain the bigger the effect of a tiny change over the whole chain
  - Slows learning down

### Residual Connections

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
    - thus, the vgg block wants to learn 0 if $h(x)$ has to be the identity, and that is very easy for optimization
    - the layer learns the residual mapping $f(x)$ and not the layer's output $h(x)$
- What is done in practice?
  - We put some convolutional layers that transform the input into a higher dimensional representation
  - Since the output has to have the dimension as the input, we add a 1x1 convolutional layer to the residual connection