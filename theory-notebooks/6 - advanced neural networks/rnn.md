# Recurrent Neural Networks

- Why?
  - Focused on learning on sequential data
  - Handle temporal constraints on the input
  - Handle variable-length data
  - Sequences are made of elements that cannot be considered statistically independent from one another
    - With images, sampling one image or another from the dataset distribution does not change anything
- How?
  - Maintains an internal state with information about the past
  - Basically adds the contribution of the past weights with the corresponding weights
