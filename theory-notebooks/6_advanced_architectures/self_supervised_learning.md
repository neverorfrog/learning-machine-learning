# Self Supervised Learning

- Learning representations of data without having an explicit labelling at hand
- How?
  - Comparing similar objects

## Siamese Neural Networks

- Two identical networks that want to compare how similar two images are
- How?
  - They encode the images into a [latent space](../2_basics_of_nn/misc_definitions.md#latent-space)
    - It is much easier to see how close two images are in that space
    - Comparison happens through dot product (acts as a decoder)
  - Encoder is the only part having parameters, forcing the network to learn the latent space through the loss (with the objective of making same images closer)
- Why?
  - To generate artificial labels
  - Acts as a pretraining phase for downstream tasks
- IMPORTANT
  - Performance of the SSL task is not important
  - Performance is evaluated on the downstream task

## Contrastive Learning

- The model is trained to bring representations of two objects of the same class closer together in an embedding space, while pairs of different classes apart.
- The typical loss considers pairs of data points and minimizes their distance in embedding space if they are of the same class
- $\mathcal{L}(x_i,x_j,\theta) = \mathbb{1}(y_i=y_j)(f_{\theta}(x_i) - f_{\theta}(x_j))^2 + \mathbb{1}(y_i \neq y_j)max(0, \epsilon - (f_{\theta}(x_i) - f_{\theta}(x_j)))^2$
  - Considering just pairs would require many pairs

### Triplet Loss

- Three elements at a time
  - Anchor (subject of comparison)
  - Positive sample (class 0)
  - Negative sample (class 1)
- Distance is minimized to positive sample and maximised to negative sample
- $\mathcal{L}(x,x_+,x_-,\theta) = max(0, (f_{\theta}(x) - f_{\theta}(x_+))^2 + \epsilon - (f_{\theta}(x) - f_{\theta}(x_-))^2)$

## Proxy Tasks

- Artificially created tasks that do not directly relate to the final goal
- Helps the model learn representations from the data
- Practically help us decide which are positive and which are negative samples
- Examples
  - Rotation, distortion
  - Relative Position of patches
    - Helps to learn relative positions (spatial features) of the input
  - Jigsaw puzzle
    - Learn relations between patches of an image

## Some tricks to have a good representation of the data

#### Heavy Data Augmentation

- Transformed version of an image is still the same image, so it will be a positive sample
- Generating new training data

#### Large Batch Size

- Allows to have many negative examples

#### Hard negative mining

- For proper learning, samples need to be very different
- So that diverse objects have very different representations
- Zebra and horse could share something

## How do we measure performance?

- Classification layer (as simple as possible) on intermediate feature layer
- Proxy tasks can be a good indicator of downstream tasks

## Some actual SSL pipelines

### SimCLR (Similarity Contrastive Learning Representation)

#### General Idea

- Contrastive Learning with classical siamese networks approach

#### Structure
- N images of a batch are fed into two same transformations each
- Each couple of transformed image is fed into a branch of two Resnets
- The latent representation is embedded (projected) by two MLPs

#### How is the loss computed?

- Every pair of augmented projected image is fed into a similarity function
  - Two views of the same image are considered as positive pairs
- These similarities are fed into a softmax function
- The cross entropy loss tries to maximise the similarity between two views of the same image

#### Drawbacks

- We need to do hard negative mining

### BYOL (Bootstrap your own latent)

#### General Idea
- Removes the need for hard negatvie mining
- Does not use a classical contrastive learning appraoch

#### Structure
- Two branches (one online network and one target network)
- Two branches have different weights
- Loss is computed between the two branches
- Loss is backpropagated online to the online network
- Target network weights are upate by polyak averaging
  - Contains the whole history thanks to the moving average

#### What is the loss?
- Cosine similarity between the two network outputs

#### Why can't we use this with siamese networks?
- Since we do polyak averaging, the loss between the two networks is never zero
  - Thus, we don't need hard negatives
  - Negatives are enforced by the batch normalization layer in the online network
- Instead, with siamese networks everything would be positive

### Barlow Twins

#### General Idea
- Builds two distorted versions of the same objective
- These version have to be similar
- This similarity is captured by a **cross-correlation matrix**
  - This needs to be near to the identity matrix

#### Structure
- Two encoders for one image that apply a distortion
- Then the embeddings 