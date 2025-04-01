# Neural Networks Possible Questions

## Datasets and Loss

1) Definition of supervised learning and dataset
    - What are the main assumptions when treating a supervised dataset?
2) Difference between supervised, unsupervised and self-supervised
3) Definition of loss function
    - Why does it need to be differentiable?
4) What is overfitting? How is the loss function connected to overfitting?
5) How is a loss function built?

## Automatic Differentiation

1) What does autodiff do?
2) Explain forward-mode autodiff
3) Explain backward-mode autodiff
4) Why is sigmoid not a good activation function?

## Linear Models for regression and classification

1) Defintion of Linear Regression
2) Why is the closed form solution of least squares not good?
3) Definition of Linear Classification
    - Why do we need softmax?
    - What happens in binary classification?
4) What is the logsumexptrick?
5) Why do we need calibration of the classifier?

## Shallow Neural Networks

1) How do you define a fully connected layer?
2) Why do we need activation functions?
3) Plot some common activation functions.
4) Define the universal approximation theorem.
5) Define gradient descent
6) What is momentum in SGD?

## Convolutional Neural Networks

1) Why are fully connected networks bad for images?
2) How do CNNs solve the problems related to fully connected?
3) Write the expression of the convolution operation.
4) What is a receptive field?
5) What is the reason for activation functions in CNNs?
6) What do we need the max-pooling layer for? What principle does it implement?
7) What do we use 1x1 convolution for?
8) Why do we need groupwise convolution and how does it work?

## Scaling up the models

1) What is explicit regularization?
2) What is early stopping and when is it useful?
3) What is data agumentation?
4) What is dropout?
    - how is connected to data augmentation?
    - why is dropout a problem a problem in inference?
    - list dropout methods at inference time.
    - when is dropout executed and why?
5) What is batch normalization?
    - benefits and drawbacks?
    - how does it work (with FC and with CNN)?
    - what happens during inference?
    - what is layer normalization? what problems does it fix?
    - how many learnable parameters?
6) What are residual connections?
    - why are they used?
    - typical design of a residual block
    - relation between resnets of models

## Convolutions beyond images

1) What about 1D and 3D convolutions?
2) How can CNNs be applied to text?
3) How to deal with long sequences?
4) What is forecasting?
5) How are causal models used in forecasting?

## Recurrent Neural Networks

1) What is a hidden variable?
2) How does a generic RNN work?
3) What are the flaws of classical RNNs?
4) Describe GRU
5) Describe LSTM
6) What about stacking RNNs on multiple layers?
7) What about bidirectional RNNs? Why are they useful?

## Transformers

1) What problem do transformers address over CNNs?
2) Explain self-attention (single and multi head)
3) What are the parameters and hyperparameters?
4) Why do we need positional embeddings?
5) Describe the transformer block
6) How could we perform classification?
7) Describe the encoder-decoder architecture
8) What makes the decoder different from the encoder?
9) What is the computational efficiency of transformers?
10) What about transformers for images?

## Generative Models

1) Describe VAE
2) Describe GAN
    - How is the loss mode?
    - Describe the training scheme (with the gradients)
    - How do we evaluate GANs?