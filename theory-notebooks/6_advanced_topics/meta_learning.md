# Meta Learning
- We want the model to adapt to different concepts and tasks, not to different data
- Similar to transfer-learning, but in meta-learning no weight modification is done like in fine-tuning

## What does it mean to train over tasks?
- It mean we from a probability distribution over datasets (not data samples)
- Each task is associated with a dataset

## What is K-shot learning?
- Training Dataset is a tuple of two subsets
  - Support set S
    - Contains labelled data (N classes)
    - Each sample is actually a tuple of K samples
  - Prediction set B
    - Labelled data that needs to be classified
    - It is not a test set
- Test set is on a totally different task
- The goal is to use the support set to actually label correctly the
  samples in the prediction set

## How do we train in meta-learning?
- We still use MLE but with a twist
  - We sample first of all a TASK
  - We sample a support set and a batch set
  - In computing the likelihoods, we use the support set
  - The model should use the support set to draw conclusions on unseen tasks
- The main goal is not to classify
  - It is to extract features to discriminate elements in the support set
- We have three main approaches:
  - Metric-based Learning
  - Optimization based
  - bla

## Metric-based Learning

### Siamese Neural Networks
- Mimicks knn learning
- We use support set as the training data
  - I just care about who is near me in the support set
- $P_\theta(y|x,S)=\sum_{(x_i,y_i)\in S} k_\theta(x_i,y_i)y_i$
  - We compute the probability of the sample being of that class by
    weighing the probability of the neighbors beign of that class, weighted
    by the distance
#### Drawbacks
- You require more shots than few shots
#### How do you learn the kernel k?
- Kernel k computes the distance between two samples
- We could use two embeddings from the two branches
  - We would need a pretrained network
- Compute distancce between them
- Apply maximum likelihood estimation as in knn
#### How do you do inference?
- The predicted class is the one corresponding to the sample closest to me

### Matching Networks
- Instead of considering the distance from a single sample
- Compute attention mask wrt the whole support set

### Relation Networks
- In relation networks we use a single encoder to obtain the embeddings for
both the support set and the new datapoint.
- After this, we concatenate the same embedding of the new datapoint to each embedding of the support set and feed this to a function that will output a relation score

## Optimization-Based Learning
- Can we learn gradient descent?
- RNN could be used
  - One optimizer computes the weights of the function that computes the gradient
  - One optimizee computes the gradient
### How do you train?
- An issue is that the actual gradient computation is non-differentiable
- The optimizer want to minimize the distance between the output of itself and $-\alpha * \nabla$