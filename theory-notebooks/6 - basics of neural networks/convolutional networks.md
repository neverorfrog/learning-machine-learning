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

- 1D Convolution
  - A linear operation where two sequences of numbers output a third sequence
  - Ingredients
    - Input sequence x, kernel w, output sequence z
    - Input and output of length m, kernel of length n
  - $(x * w)_n[p] = \sum_i x_{p+i} \cdot w_{n-i}$
    - The p-th element of the convolution is a dot product
  - Intuitively, flipping the convolver *b* and sliding it from left to right along convolvee *a*

- 2D Convolution
  - What is the result? What are the inputs?
  - What is the operation?

- In any case, the same set of weights is used everywhere

## Why are convolutions useful

- Images (and other signal type data) have some properties that bias the learning process in some way
  - Translation invariance: an object remains the same if it is in different regions
  - Locality: an object must be recognized independently of what is around
- Depending on the task there might be other properties that bias the learning process

## Convolutional Layer

- Input : image of dimension (c,w,h) = (channels, width, height)
- Parameters : filters (learnable) usually much smaller than the image itself
- Output : a 3D map of features of dimension (c,w,h) where c is the number of channels (features) we want to learn, while (w,h) is the dimension of the image after the convolution operation

### Local receptive field

- Every hidden unit looks at a specific region of the data
- For example, in a 28x28 image, the LRF could be a 5x5 region
- **Stride** length is how much we shift the LRF from one unit to the other
- On each layer we iterate over alle the LRF

### Shared weights and biases

- The same set of wandb is associated to every LRF in the same layer
- This means every hidden unit detects the same feature in each LRF
- **Feature map** is what lies between the input layer and the hidden layer, the shared wandb are called **kernel**

### Pooling

- Layer after the convolutional to condense what comes out of it
