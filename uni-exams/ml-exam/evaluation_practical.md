# Evaluation and practical problems

## Overfitting

- Provide a formal and general definition of overfitting, without referring to any particular model
  - Premise: we have two learning algorithms, a test set T and a training set
      S. Moreover we have an estimator for the error between the learned
      function and the true function. Overfitting describes a situation in which
      $error_1(T) > error_2(T)$ and $error_1(S) < error_2(S)$. Specifically,
      algorithm 1 is said to be overfitting on training data, because it behaves
      better on training data, but worse on test data than algorithm 2
  - More intuitively, it indicates the fact the hypothesis space is too expressive and too powerful, thus capable of catching too many features of the training data, while potentially ignoring features of data never seen before.
  - Overfitting hinders generalization
- Show two examples of overfitting in two distinct models
  - Linear Regression
  - Decision Trees
- For one of the models above, explain how the problem can be mitigated
