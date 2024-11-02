# Gaussian Mixture Models

- Dataset is without labels
- How do represent data?
  - We can represent the dataset by a mixture of gaussians
  - $P(x) = \sum_k \pi_k N(x; \mu_k, \Sigma_k)$
- Given unlabeled data, how can we reconstruct $P(x)$, namely estimate the parameters of the $k$ gaussians?

## K-Means Algorithm

- Estimate the means of the gaussians
- Iterative algorithm
  - For every data point take the nearest centroid
  - Move the centroids according to the "registered" data points by taking the average distance
  - Repeat until convergence
  - Converges always but depends heavily on initialization
- Comments  
  - Fixed covariance with euclidean distance (so only circle shaped stuff is recognized)
  - Very sensitive to outliers

## Latent Variables

- Definition
  - Each sample $x$ has a $k$-dimensional variable $z$ where $z_k=1$ if the $k$-th gaussian generated that sample
  - So $P(z_k=1|x) = \pi_k \rightarrow$ the bigger the gaussian, the bigger $\pi_k$

- Usage
  - The goal is to estimate $z$ for each data point $\rightarrow P(z) = \Pi_k \pi_k ^{z_k}$
  - In the same way $P(x|z) = \Pi_k N(x;\mu_k,\Sigma_k) ^{z_k} \rightarrow$ given z, x depends only on the gaussian

- Connection to GMM
  - We can marginalize out $z$, getting thus $P(x) = \Sigma_z P(x|z)P(z) = \sum_k \pi_k N(x; \mu_k, \Sigma_k)$
    - GMM can be seen as marginalization over z of $P(x,z)$
  - We can also apply Bayes Rule, getting posterior $P(z_k=1|x) = \frac{P(z_k)P(x|z_k=1)}{P(x)} = \frac{\pi_k N(x;\mu_k)}{\sum_k \pi_k N(x; \mu_k, \Sigma_k)}$

## Expectation Maximization

- The goal is to be able to express $P(x)$ as function of the parameters $\pi, \mu, \Sigma$ for each datapoint
- ML principle: $argmax P(x|\pi, \mu, \Sigma)$ we find expressions of these parameters in function of $x$ and the posterior $P(z_k=1|x)$
- Since we don't know neither the parameters, nor the posterior, we iterate after initializing the parameters
  - **Expectation Step**: compute posterior from estimated parameters (sort of assigning centroid to datapoint)
  - **Maximization Step**: compute parameters from posterior and dataset by maximising the likelihood (sort of realibrating centroid position and covariance)
- Generalization of K-Means
  - We compute also the posterior probability for each datapoint and not just a one hot encoded assignment
  - We generalize more by using also the covariance
- Generalizable to any distribution
  - $\theta$ can be the parameters of the model and some latent variables to denote a datasample
  - We may hop from parameters to latent variables in an iterative scheme until convergence (maximising log-likelihood)
- Comments
  - Outliers don't create problems in correctness, but slow down the convergence
  - Initialization may cause to find only a local maximum
