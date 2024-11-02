# Motion Estimation with CNNs

## FlowNet

- What?
  - Given two consecutive frames, match features at different locations to find if an object is in a different location in the second image
- How?
  - Encoder-decoder architecture
  - Encoder does downsampling through multiple convolutional layers, and thus extracts features
    - FlowNetSimple: images are processed together in a single convnet
    - FlowNetCorr: images are processed separately and their features are combined together later with a correlation layer
    - In both cases, features are skip-connected to the decoder to ensure smooth optimization
  - Decoder scales image to original resolution with upconvolutional layers and also combines these outputs with the features from the contractive layers

## PWCNet

- Feature Pyramids
- Cost volume for flow estimation
