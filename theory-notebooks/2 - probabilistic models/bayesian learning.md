# Bayes Theorem

- Bayes theorem: $P(h|D) = \frac{P(D|h)P(h)}{P(D)}$
  - $P(D)$: prior probability of observing a data sample (input and label in case of supervised learning)
  - $P(h)$: prior probability of an hypothesis h (output of our model)
  - $P(h|D)$: posterior probability
  - $P(D|h)$: likelihood, probability that the hypothesis reflects the data

## Maximum a Posteriori Hypothesis

- $h = \argmax_h P(h|D)$
- Several learning methods use this principle

### Brute Force MAP Hypothesis Learner

- We select a bunch of hypotheses and compute their posterior probability and then select the one with maximum posterior probability
- The problem with that is that the most probable hypothesis may not correspond to the most probable classificatione

### Optimal Bayes Classifier

- We have a target function f: X -> Y with Y = {1,2,3,...,n}
- The goal is to assign a class to a new data sample $x$
- The most probable classification of the new instance is obtained by combining
  the predictions of all hypotheses, weighted by their posterior probabilities
- $f(x) = \argmax_{y_i} P(y_i|x,D) = \argmax_{y_i} \sum_{h} P(y_i|x,h)P(h|D)$
- Practically impractical, unless we have a small hypothesis space

### Naive Bayes Classifier

- Approximation of optimal bayes classifier
- We suppose sample $x$ is made of $J$ attributes
- $P(y_i|x,D)$ can be rewritten as $P(y_i|a_1,...,a_J,D)= \alpha P(a_1,...,a_J|y_i,D)P(y_i|D)$ through the Bayes rule
- Naive Bayes Assumption exploits conditional independence
  - $P(a_1,...,a_J|y_i,D)=\Pi P(a_j|y_i,D)$

#### GNB on Text

- TODO

## Maximum Likelihood Hypothesis

- The data obeys to a probability distribution (likelihood) $P(y_n|x_n;\phi)$
- The hypothesis space is continuous (parametrized) in $\phi$
- The goal is to find the $\phi$ that maximises the likelihood of that dataset given these parameters
- $h_{ML} = \hat \phi = \argmin_{\phi} -\log(\Pi_n P(y_n|x_n;\phi))$
- Practical only if we have enough data
