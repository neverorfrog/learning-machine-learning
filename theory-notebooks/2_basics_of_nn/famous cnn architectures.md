# Famous CNN Architectures

## LeNet

- Most basic one
- Just convolutional, pooling and sigmoid

## AlexNet

- Exploits the concept of hierarchical features
- Exploits deeper architecture
- Introduces normalization layers
- Uses relu instead of sigmoid

## VGGNet

- Modular architecture
- Found that deep and narrow outperforms shallow and wide
- One module is made of
  - 3x3 kernel with padding 1
  - relu activation
  - 2x2 pooling kernel
- Exploited parallel GPUs

## GoogLeNet

- Uses inception blocks (parallele convolutions)
- Avoids fully connected layers

## ResNet

### Residual Connections

- What?
  - $h(x)=f(x)+x$ where $f(x)$ is the function applied by the VGG block
  - So, the input contributes directly to the output
- Why?
  - Avoids shattered gradients
    - In the early stages of learning gradients can change drastically depending on the input
    - If the input contributes directly to the gradient, it was found that it happens less
  - Tending towards nested function space
    - residual block before relu learns $f(x)=h(x)-x$
    - nested function space means that each layer tends towards the idendity function
    - thus, the vgg block wants to learn 0 if $h(x)$ has to be the identity, and that is very easy for optimization
    - the layer learns the residual mapping $f(x)$ and not the layer's output $h(x)$
- What is done in practice?
  - We put some convolutional layers that transform the input into a higher dimensional representation
  - Since the output has to have the dimension as the input, we add a 1x1 convolutional layer to the residual connection
