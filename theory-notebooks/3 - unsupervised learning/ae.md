# Autoencoders

## What is the goal?

- Same problem of PCA, but with a nonlinear latent space
- We want to obtain a lower-dimensional representation (latent variables) of some data

## What is the general approach?

- Trained by using the same input in input and in output
- Two networks: encoder and decoder
  - Encoder takes as input the data $x$ and outputs the low-dim represnetation $z$
  - Decoder takes as input the the low-dim represnetation $z$ and outputs the original data $x$
- We train them together based on reconstruction loss, but we could then use the two networks on their own
  - Reconstruction loss $L(x,\hat x)$ is MSE between input and reconstructed input
- Example: $3D \rightarrow 2D \rightarrow 3D$
  - Input data in 3 dimensions (volume)
  - Latent variables in 2 dimensions (surface)
  - Output data is in 3 dimensions, **but** we don't get a volume as in the input data, instead we get a manifold as representative as possible of the original volume

## Autoencoders for anomaly detection

- Problem is that we want to learn a function
$f: X \rightarrow \{NORMAL,ABNORMAL\}$ while  having only $NORMAL$ examples in the training data
- Solution Template
  - Train an autoencoder as if the dataset were unlabeled (since we have just one class)
  - Compute a certain treshold $\delta$ based on the loss (e.g. the mean)
  - Do inference with autoencoder on $x'$, computing a $loss'$ and classify it as $NORMAL/ABNORMAL$ based on $\delta$
- Also, more generally the goal could be, in classical binary classifcation, to consider a third option "don"t know" if the image is neither of class $A$, nor $B$
