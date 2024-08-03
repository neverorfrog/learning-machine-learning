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

## Performance metrics

- Discuss the statemenet: Accuracy is not always a good performance metric for classification
  - This is true, especially if we have an unbalanced dataset
  - For example, let's say we are in a binary classification setting, with dataset made of 900 positive samples and 100 negative samples
    - Let's suppose we have some stupid algorithm that alway predicts positive
    - With this dataset the accuracy would be 90%, which would be reasonable if the dataset was balanced
    - But obviously a previously unseen sample will be classified based on no processing of the features and based on the fact that the accuracy was satisfying

- What are some alternatives to accuracy for unbalanced datasets?
  - Precision, recall, F1-score and confusion matrix
  - Recall is defined as the ability avoid false negatives
    - $R=\frac{TP}{FN+TP}$
    - Low recall $\righarrow$ many false negatives
  - Precision is the ability to avoid false positives
    - $P = \frac{TP}{FP+TP}$
