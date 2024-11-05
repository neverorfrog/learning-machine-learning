# Convolutional Neural Networks

### Phases of a convolutional neural network:
  - Feature extraction (encoding of the image)
  - Classification (decoding of the class by fully connected layers)

### Why using just Multilayer Perceptrons is bad for images (or audio)
- Dimensionality
  - Images can be very big and this would cause an explosion of the number of parameters
- Training
  - Local minima
  - Exploding and vanishing gradients due to dense layers

### How can we define a different kind of layer?
- We need to exploit spatial information (nearby pixels are probably related)
- Fully connected layers flatten the image, losing ordering of pixels

### What properties can we capture in images and audios?

- Self-similarty: patterns are repeated across [patches](misc_definitions.md#patch)  Maybe we could share the weights?
- Locality: an object must be recognized independently of what is around
- Translational Invariance: the representation of an object is independent of where it is
- Deformation Invariance: to some extent the representation of an object is independent of how much it is deformed

**These properties bias the learning process**

## Convolution (Discrete)

- Exploits the fact the far pixels don't have influence
- Thus, we develop a local layer that acts only on the patch
- Local layer: $[f(X)]_{ij} = f(P_k(i,j))$
  - $X$ is the image
  - $P_k(i,j)$ is the patch
- Also expressed as multiplication of a matrix *C(w)* by the input. Turns out this matrix is circulant
  - It is [SHIFT EQUIVARIANT](misc_definitions.md#equivariance)
  - And also commutative wrt multiplication with another circulant matrix

### How?

- A linear operation where two arrays of numbers output a third one
- Ingredients
  - Input image x, kernel w, output image z
  - Intuitively, flipping the convolver *w* and sliding it from left to right along convolvee *x*
- Mathematical 2D Convolution
  - $x[i,j]=\Sigma_{k,l} (w[k,l]x[i-k,j-l])$
- **Stride** length is how much we shift the LRF from one unit to the other
- **Padding** is a way of filling pixels beyond the border, such that we can convolve also at the border of the image

### Receptive Field

- Subset of an input X that contributed to the output of a convolutional model
- For a single layer, the receptive field is a patch $P_k(i,j)$

## Convolutional Layer

- Input : image of dimension $(c,h,w)$ = (channels, height, width)
- Parameters : filters (learnable) usually much smaller than the image itself
  - $W \sim (c',ssc)$
- Output : a 3D map of features of dimension $(c',h',w')$ where 
  - $c'$ is the embedding size (number of features maps we want to learn)
  - $(h',w')$ is the dimension of the image after the convolution operation
- Operation: convolution + bias summation over the whole image with the same kernel
- Learning proces: learn the kernel in order to extract features

### Properties

- **Locality**: the kernel considers only local information, supposing the features are translation invariant
  - Every hidden unit looks at a specific region of the data
  - For example, in a 28x28 image, the kernel could be a 5x5 region
- Translation equivariance: an object remains the same if it is in different regions
    - $P_k(i,j) = P_k(i',j') \implies f(P_k(i,j)) = f(P_k(i',j'))$
- **Parameter sharing**: the same sets of weights is used everywhere
  - This means every hidden unit detects the same feature in each local receptive field
  - **Why?**: to achieve translation equivariance
- **Mathematically**:
  - TODO
- **How many parameters**?:
  - TODO

## Other useful operations in CNNs

### Downsampling (Pooling Layer)

- More we go forward in the network, more we are downsampling the image
  - lrf becomes bigger relatively to the image
  - scaling down representation size
- Why?:
  - **Hierarchy of features**: early layers will learn lower level features, so we need to aggregate the features while we traverse the neural network
  - The network becomes more sensitive to translation while we go forward
  - We are squeezing information out and distilling it from noise
- What?
  - Kernel with no learnable parameters
  - Typically outputs maximum or average value over lrf
- In the end
  - We downsample the image and thus combine information of adjacent pixels
  - We want to combine low-level features to get higher-level features

### Upsampling

- Why?
  - Image reconstruction (usable for segmentation)
- How?
  - Static
  - Zeros
  - Bilinear Interpolation
  - Transposed convolution
    - In normal convolution an output activation is the weighted sum of $K^d$
      (where $d$ is the dimension of the data sample) input units (pixels)
    - Transposed convolution is dual: one input unit generates $K^d$ output
      activations 

### Changing the number of channels

- Why?
  - TODO
- How?
  - 1x1 convolution


## Famous CNN Architectures

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

- Uses inception blocks (parallel convolutions)
- Avoids fully connected layers
