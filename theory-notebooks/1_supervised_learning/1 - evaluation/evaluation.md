# Evaluation of ML models

## True vs Sample Error

- True error $E_D(h)=P_{x \in D}[f(x) \neq h(x)]$ is not measurable in practice because we don't know the underlying probability distribution $D$, otherwise learning would be useless
- Instead, we use a sample error $E_S(h)=\frac{1}{n}\delta_{x \in S}(f(x) \neq h(x))$ to estimate the true error
- We achieve an unbiased estimate of the error by using a training set for hypothesis h and a test set for sample error

## Bias/Variance Tradeoff

### Bias

- Bias indicates how much an estimator of a value is shifted away a priori from its true value
  - $bias(\hat \theta _m) = \mathbb{E}(\hat \theta _m) - \theta$
  - An estimator is simply a function of the data
- The estimator is unbiased if bias = 0
  - For example, the sample mean of a gaussian distribution is unbiased

### Variance

- variance indicates how much an estimate changes if we resample again from the same distribution
  - $var(\hat \theta)$

### In the end

- Both bias and variance should be low
- We can achieve this by lowering the mean squared test error, which incorporates both bias and variance

## K-fold cross validation

- Answers to two questions: How can we lower the confidence intervals of the sample error? How can we compare two learning algorithms?
- Procedure
  - Partition dataset in K subsets $S_k$
  - For each subset
    - Take $S_k$ as test set and the others as training set
    - Compute $E_{S_k}(h)$
  - Compute $E=\frac{1}{k}\Sigma E_{S_k}(h)$
- If we want to compare two algorithms we simply compute $E_{S_k}(h)$ as the difference between two hypotheses