# Probabilistic interpretation of Learning

## Maximum likelihood hypothesis

- Provide a formal deﬁnition
  - Premise: we are in the context of parametric models, where the goal is to learn a function $f$ with a model $\phi = g[x,\theta]$. The input is $x$, $\theta$ are the parameters of the model and  $\phi$ are the parameters of the probability distribution $P(y|x;\phi)$
  - This probability distribution tells us how likely a value y is, given the input x, based on the parameters $\phi$. This probability distribution is the likelihood function.
  - Given this information, the maximum likelihood hypothesis states that maximising this likelihood function

## Maximum a Posteriori hypothesis

- Provide a formal definition
  - $h^* = \argmax_h P(h|D)$
- Provide a formal definition of Optimal Bayes Classifier
  - $y = \argmax_{y_k} \Sigma_h P(y_k|h)P(h|D)$

## Secret strings

- A secret string s = $b_0b_1b_2b_3$ of 4 bits fulﬁlls the following constraints
  - if $b_0 = 0$, s contains an even number of 0’s and 1’s
  - if $b_0 = 1$, s contains at least three 1’s

- Deﬁne the prior probability distribution $P(s)$ of the hypothesis string s
  - $P(s)=\frac{1}{7}$
- Assuming $b_0 = 0$, deﬁne the conditional probability distribution $P(s|b0 = 0)$ and indicate all maximum
a-posteriori hypotheses
  - $P(s|b_0 = 0) = P(b_0 = 0|s)P(s)=\frac{3}{7}\frac{1}{7}$
  - The MAP hypotheses are the ones that are possible when the first bit is 0
- Assuming b0 = 1 and b1 = 1, indicate all maximum likelihood hypotheses and compute the likelihood
that b2 = 1

## Generative model

Consider a dataset D for the binary classification problem $f: \R^3 \rightarrow \{A,B\}$

- Describe a probabilistic generative model for such a classiﬁcation problem, assuming Gaussian distributions
  - In a generative model for classification, we predict the class by computing the posterior probability distribution $P(C_i|x)$ from the class-conditional densities $P(x|C_i)$
  - This means the model first learns how to generate a sample $x$ given it has to be of a certain class
    - From Bayes rule: $P(C_i|x)=\frac{P(x|C_i)P(C_i)}{P(x)}=\frac{ln(P(x|C_i)P(C_i))}{\Sigma(ln(P(x|C_j)P(C_j)))}=\frac{a_i}{\Sigma a_j}$
    - Normalizing (softmax): $P(C_i|x)=\frac{exp(a_i)}{\Sigma exp(a_j)}$
    - We assume that $P(x|C_i)$ is a normal distribution with mean $\mu_i$ and covariance $\Sigma$ and that the prior $P(C_i)=\pi_i$, which has two implications:
      - $a_i$ becomes linear in $x$ and the $w$ vector obtained with the gaussian's parameters (that's true for all exponential class-conditional densities)
      - gaussian's parameters of $P(x|C_i)$ obtained through maximum likelihood estimation
  - In the end the sample $x$ is classified by taking the maximum among $P(C_i|x)$
- Identify the parameters of the model and determine the size of the model (i.e., the number of independent
parameters)
  - The parameters are those of the gaussians
  - The size is in total 13:
    - 1*1 for $\pi_i$
    - 2*3 for $\mu_i$
    - 6 for $\Sigma$
