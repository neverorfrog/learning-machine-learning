# Artificial Neural Networks

## Backpropagation

- Describe Backpropagation
  - We suppose the network is made of $L$ layers
  - Backpropagation per se is not a learning algorithm, but just a smart way of propagating the gradient of the loss through the whole network. It happens in two phases: forward and backward pass:
  - Forward pass:
    - This is where the loss is actually computed through inference
    - Initialization with $a^{[0]} = x \in \R^{d+1}$ 
    - For each layer $l$ from 1 to $L$
      - $z^{[l]} = W^{[l]}a^{[l-1]}$ pre-activations
      - $a^{[l]}=f(z^{[l]})$ where $f$ is the activation
    - At the end $y=a^{[L]}$ and $L = L(t,y)$
  - Backward pass:
    - This is where the gradient of the loss is computed through iteratively applying the chain rule
    - The output will be a derivative with respect to the weights for each layer, because ultimately the goal is to shift the weights toward the negative gradient of the loss
    - Initialization with $g= \nabla_y L(t,y)$
    - For each layer $l$ from $L$ to $0$
      - $g = g \cdot f'(z^{[L]}) \rightarrow$ propagate gradient to pre-activation
      - $\nabla_{w^{[l]}} = g (a^{l-1})^T$
      - $g= (W^{[l]})^T g \rightarrow $ propagate gradient to previous hidden layer
- Is it affected by local minima? How can we attenuate it?
  - Backpropagation is not affected by it, but SGD (which backpropagation is a parto of) is. Local minima happens when the learning rate is not adequate or the initialization is unfortunate, so we could change these hyperparameters to avoid it.
- Is it affected by overfitting? How can we attenuate it?
  - Neither this one is a problem of backpropagation, since it is not a learning algorithm. Overfitting affects the learning algorithm itself, namely SGD.
