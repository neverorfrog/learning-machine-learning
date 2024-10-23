# Variational Autoencoders

- Plain autoencoders are good to do dimensionality reduction
- **BUT** there is in general no direct implication from distance in latent space to similarity in input space
  - If two samples are near in the latent space, this does not mean they are similar in input space
  - There are no constraints imposed in how the latent space is learnt
  - $\implies$ autoencoders are not good to generate samples from data in latent space

## Concept

- Encoder produces a gaussian distribution instead of a simple vector
  - In practice the output is the mean and covariance of the gaussian distribution representing the latent space
- Decoder operates on samples from this distribution
- We also change the loss function by adding a KL divergence factor
  - Distance from current latent space distribution $N(z;\mu,\Sigma)$ to a nominal distribution $N(z;0,I)$
  - This distance is also weighted by a weight $\lambda$
- Practical problem: z sampling is not differentiable
  - Solved by reparametrizing
  - Sampling is done by $z=\mu+\Sigma \epsilon$ where $\epsilon$ is sampled from an "external" default distribution
