# Gradient Descent

An optimization problem is simply defined as $x^* = argmin f(x)$

## How do we define mathematically gradient descent?

- $x_t = x_{t-1} + g_t$
- $g_t = -\eta_t\nabla f(x_{t-1}) + \lambda g_{t-1}$
  - The gradient of f is a vector of partial derivatives. It points in the direction of the steepest ascent of the function
  - The second term is momentum

## Optimization in the context of ML?

- We have a function approximation that yields a $\hat y_i$ for every datapoint $x_i$
- We can define a cost function as a sum of losses for every datapoint
- Cost needs to be minimized iteratively: gradient descent and autodiff

### What is momentum?

- It conserves some direction from the previous gradient iteration
- Momentum can be shown to accelerate training by smoothing the optimization path

## Loss Function

- In minibatch gradient descent, we define the loss as the mean of losses on all datapoints of the minibatch
- To find the optimal parameters, we will minimize this loss

### Differentiability

- Differentiability of this function is important, otherwise we remain stuck
- Also, we will use autodiff

### Expected Risk

- Our true objective is minimizing some average loss on some unknown, future input yet to be seen
  - The elements of our training set are only a proxy to this end
  - It's like we try to minimize the expected value of the loss over the whole data distribution, namely across all the possible input-output pairs the model could see
- **However**, this quantity is impossible to compute
  - Thus, we compute the **empirical risk** as a proxy, using the training dataset
  - The difference in loss between the expected and the empirical risk is called the **generalization gap**

