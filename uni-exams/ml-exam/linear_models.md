# Linear Models

## Linear Regression

### Dataset of tuples $(x_n, t_n)$ where $x_n \in R^3$ and $t_n \in R$

- Provide the definition of linear regression with parameters $w$ for estimating a non-linear function $y$
  - $w$ is 4-dim (bias in pos 0)
  - Inference: $\hat y(x)=\Sigma_k w_k\cdot \phi(x_k)$ with $k = 0,1,2,3$
- Provide a suitable loss function and sketch an algorithm for estimating the parameters of the model
  - $E(w)=\frac{1}{2}\Sigma_n (y(x) - \hat y(x))^2$
    - We could also add regularization term $E_w(w) = \frac{1}{2}w^tw$ to $E(w)$ to penalize high weights and avoid overfitting
  - Sequential learning, namely following the negative gradient
    - $w \leftarrow w + \eta \phi(x_n)(y(x_n)-\hat y(x_n))^2$
    - Heavily depends on learning rate
