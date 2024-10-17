# Sequential Learning

- We partition the dataset and the models are trained in sequence
- While in bagging, the bootstrap dataset selection happens in binary way, namely a sample is or not is in the subset, here this is what happens iteratively:
  - The vector of random weights for sampling is made of real values (at first all equal (uniform sampling))
  - We train a model on a bootstrap set with these weights
  - Based on the performance of this classifier, weights corresponding to incorrectly classified samples are increased
- But how exactly do we use the weights at each model training of the sequence?
  - In the error function by weighting each misclassified sample
- And how exactly are these weights updated?
  - Adaboost