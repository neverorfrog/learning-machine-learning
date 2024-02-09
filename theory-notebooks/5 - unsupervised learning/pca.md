# Principal Component Analysis

- Motivation
  - Data live on a lower dimensional manifold
  - We seek for lower dimensional embedding that loses smallest possible information

## Definition

- Elements:
  - Input space defined by $\{x_n\} \in R^D$ with basis $x_1,...,x_D$
  - Output (lower-dim) space in $R^M$ with basis $u_1,...,u_M$
  - We project $x_n$ into output space, we obtain a $M$-dim vector $<u_1 x_n, ..., u_M x_n>$

- Problem:
  - Choose the basis of the output space such that the variance of the projected vectors is maximised

- Procedure:
  - Define data-centered matrix $X$ (mean of the data is in the origin now, removing bias)
  - Define variance of points as the outer product of the distances between each point and the mean
    - $S=\frac{1}{N} \Sigma_n (x_n - \bar x)(x_n - \bar x)^T = \frac{1}{N} X^T X$
  - If we are projecting on $u_1$ the projected variance is $u_1^T S u_1$
  - We maximise this: $\max u_1^T S u_1 +\lambda_1(1 - u_1^Tu_1)$
  - The solution for that is $Su_1 = \lambda_1 u_1$, namely $u_1$ is an eigenvector for $S$ corresponding to the highest eigenvalue $\lambda_1$
  - That is the first principal component

- In general:
  - We take the first $M$ highest eigenvalues, giving $M$ eigenvectors and get the best basis for projecting the dataset onto a lower dimensional space

- Important:
  - **PCA does a linear transformation**

## PCA for High-Dimensional Data

- Situation in which $N < D$
- We simply take the covariance with the product $XX^T$ instead of $X^TX$ and the eigenvalues don't change
- To find the eigenvectors we left-multiply by $X^T$

## Probabilistic PCA

- We want to represent data $x \in \R^D$ in a reduced space by the variable $z \in \R^M$ assuming a linear relationship from $x$ to $z$
- We also assume that $z$ obey to gaussian distribution
- This is a generative model, because :
  - We estimate the parameters of the gaussians of $z$ through maximum likelihood
  - We generate $x$ with conditional density $P(x|z)$
