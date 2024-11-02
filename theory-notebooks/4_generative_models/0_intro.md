# Generative Models

## What difference is there between discirminative and generative?
- Discriminative learns conditional probability $P(y|x)$
- Generative tries to learn $P(x)$
  - Estimate the class-conditional densities $P(x|C_i)$
    - Instead of assigning to an image the probability of being cat or dog, genmodels learn the features of cats and dogs, generate a sample relative to a dog and check if the to-be-classified sample has similar enough features
    - We wanna be able to sample $x$


## What properties should a generative model have?
- Efficient and reliable sampling (from the latent space)
- Coverage (samples should represent the entire trianing distribution)
- Smooth changes in $z$ correspond to smooth changes in $x$
- Efficient likelihood computation

## How do we measure performance of generative models?
