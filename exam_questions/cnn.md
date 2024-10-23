# Convolutional Neural Networks

- Describe the typical stages of a single convolutional layer in a CNN
  - Convolution (feature extraction)
    - In this stage we let the kernels slide over the image to extract some features
    - Some characteristics
      - Sparse connectivity
      - Parameter sharing
      - Padding
  - Nonlinear activation (feature detection)
    - Detects valid features for the task to be learned
  - Pooling
    - Has no learnable parameters
    - Implements invariance to local translations (introduces inductive bias)
