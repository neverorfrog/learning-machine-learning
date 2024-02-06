# Linear Classification

## What does it mean when data is linearly separable?

- It means there exist a linear discriminator function that separates samples of different classes
- Mathematically, we define the linear discriminant as: $y(x) = w_k^T x$ where $k$ goes from 1 to n (number of classes) (we assume the weight vector already contains the bias)
- For every sample $x$, we assign the class $C_k = \argmax_k  w_k^T x$
- The discriminant between class k and j is a hyperplane in $d-1$ dimensions ($d$ is the number of features) expressed as $(w_k - w_j)^T x = 0$

## Problem Definition

- Target function $f: X \in \R^d \rightarrow \{C_1,...,C_k\}$
- Dataset $D = \{x_m, t_m\}$
- Inference based on linear function $y(x)=w^Tx$
- We need to find some optimal $w*$ that minimize an error function
- Example: binary classification
  - Target function $f: X \rightarrow \{+1,-1\}$
  - Just one weight vector $w$, thus $y(x) = w^Tx$
  - We can deduce that: $t_my(x_m)>0$  $\forall m$
  - Distance from data point to discriminator: $\frac{|y(x)|}{||w||}=\frac{t_my(x_m)}{||w||}$

## Some linear classifiers

### Least Squares

- One-hot encoding: label $t_m$ is a $n$-dim vector where only the $k$-th element is 1 and the others 0
- $J(w) = 1/2Tr[(y(x)-t_m)^T(y(x)-t_m)]$
- Minimize this error by taking pseudoinverse
- Comments
  - Closed form solution
  - Not resistant to outliers

### Perceptron

- Comments
  - Iterative scheme
  - Sensitive to initialization and learning rate

### SVM

- Aims at finding the maximum margin for better accuracy
- What is a margin?
  - Distance from discriminant to closest data point $x$, with distance defined [here](#problem-definition)
  - Putting problem in canonical form
    - $t_m y(x_m) = 1$ for the closest point
    - $t_m y(x_m) \geq 1$ $\forall m$
  - The margin is then $\frac{1}{||w||}$
- Thus, we can define the optimization problem:
  - $w^*=\argmax_w (\frac{1}{||w||}) \rightarrow$ maximising margin
  - subject to $t_m y(x_m) \geq 1$
- Solved with lagrangian multipliers
  - Ok
