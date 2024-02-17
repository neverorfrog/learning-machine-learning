# Convolutional Neural Networks

## Why Multilayer Perceptrons are bad for images (or audio)

- Dimensionality
  - Images can be very big and this would cause an explosion of the number of parameters
- Training
  - Local minima
  - Exploding and vanishing gradients due to dense layers

How can we define a different kind of layer?

- We need to exploit spatial information (nearby pixels are probably related)
- Fully connected layers flatten the image, losing ordering of pixels

## Convolution (Discrete)

### Why?

- Images (and other signal type data) have some properties that bias the learning process in some way
  - Translation invariance: an object remains the same if it is in different regions
  - Locality: an object must be recognized independently of what is around
- Depending on the task there might be other properties that bias the learning process

### What?

- A linear operation where two arrays of numbers output a third one
- Ingredients
  - Input image x, kernel w, output image z
  - Intuitively, flipping the convolver *w* and sliding it from left to right along convolvee *x*
- Mathematical 2D Convolution
  - $x[i,j]=\Sigma_{k,l} (w[k,l]x[i-k,j-l])$
- **Stride** length is how much we shift the LRF from one unit to the other
- **Padding** is a way of filling pixels beyond the border, such that we can convolve also at the border of the image

## Convolutional Layer

- Input : image of dimension (c,w,h) = (channels, width, height)
- Parameters : filters (learnable) usually much smaller than the image itself
  - **Parameter sharing**: the same sets of weights is used everywhere
    - This means every hidden unit detects the same feature in each local receptive field
  - **Locality**: the kernel considers only local information, supposing the features are translation invariant
    - Every hidden unit looks at a specific region of the data
  - For example, in a 28x28 image, the kernel could be a 5x5 region
- Output : a 3D map of features of dimension (c,w,h) where c is the number of channels or features maps(features) we want to learn, while (w,h) is the dimension of the image after the convolution operation
- Operation: convolution + bias summation over the whole image with the same kernel
- Learning proces: learn the kernel in order to extract features

## Pooling Layer

- Note: ore we go forward in the network, more we are downsampling the image, which means the lrf becomes bigger relatively to the image
- Why?:
  - **Hierarchy of features**: early layers will learn lower level features, so we need to aggregate the features while we traverse the neural network
  - The network becomes more sensitive to translation while we go forward
- What?
  - Kernel with no learnable parameters
  - Typically outputs maximum or average value over lrf
- In the end
  - We downsample the image and thus combine information of adjacent pixels
  - We want to combine low-level features to get higher-level features

## Batch Normalization Layer

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
