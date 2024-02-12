# Multiple Learners

## Boosting

- Provide the main features about boosting
  - Ensemble learning method, where many learning algorithms are used together for the same learning problem
  - Used mainly to enhance generalization capacity, overcoming overfitting
  - It is a sequential learning method, which means that the different learning algorithms operate one after the other
  - In boosting, the concept of weighted random sampling fot creating a bootstrap dataset is central:
    - This procedure consists in "extracting" a subset from the dataset, based on some weights
  - Procedure for boosting:
    - At first the weights are all the same (uniform sampling)
    - After extracting the first bootstrap dataset, the first model is trained on it (possibly the worst)
    - The next bootstrap dataset will have different weights based on the performance of that model, namely the weights relative to samples which where misclassified will be higher
    - This process is iterated until a termination condition

## Boh

- Assume you have 4 image classifier with medium-good clasification accuracy
  1. Descirbe an ensemble method for achieving higher classification accuracy by combining such classifiers
  2. Are there any specific properties that each classifier has to have to achieve higher accuracy? If the answer is positive, explain which theere properties are.
- A possible ensemble method to use here could be bagging
- Bagging consists in training multiple learning methods in parallel on differents portions of the dataset
- How are these dataset portions generated?
  - They are called bootstrap dataset and they are generated randomly
- The important thing about bagging is that neither one of the models sees the entire dataset, which means that overfitting possibilities are reduced
- After having them all trained, their results are combined through a voting scheme
  - For example a weighted average
