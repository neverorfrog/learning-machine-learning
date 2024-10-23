# Recurrent Neural Networks

- What?
  - Networks that learn on [time series](definitions.md/#time-series)
- Context: Autoregressive Modeling
  - We want to efficiently learn $p(x_t|x_{t-1}, x_{t-2},...)$
    - Elements of a sequence are not iid
    - We want to infer the underlying data distribution
  - We don't need the entire sequence to predict the next sample
    - If we take just the previous sample/token, we talk about Markov Models
- How?
  - [LATENT](definitions.md/#latent-variable) variables
    - Internal state $h_t$ with information about the past
  - Recurrent edge: sends the output back as an input
- Where are they used?
  - [Forecasting](definitions.md/#forecasting)
  - Clustering and classification
  - Language modeling

## Main Characteristics 

### Hidden State

- Summarizes past information
- In general computed using the past hidden state: $h_t = f(x_t, h_{t-1})$

### How do you compute the loss?

- You compute the loss at each timestep t and sum each of these losses
- The backpropagation happens through time

### Forward

$H_t = \phi(W_h \cdot H_{t-1} + W_x \cdot X_t + b)$
- $W_x$ is used to transform the input vector at each timestep into the hidden state space
- $W_h$ is used to carry into the current hidden state the previous hidden state

### Numerical Instability

- Since sequences can be long, gradient clipping must be used to prevent gradient exploding or vanishing

## Some task categories

### Seq2Seq

- We can have a sequence of arbitrary length in input and of arbitrary length in output
- A general architecture for these kind of tasks is the ENCODER-DECODER
  - The encoder brings the input (at time t-1) in a specified embedding and also the hidden state at time t
  - The decoder computes the output relative to the specific task

#### Autoregressive Models (Important type of Seq2Seq)

