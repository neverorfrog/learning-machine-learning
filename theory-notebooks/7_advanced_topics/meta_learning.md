# Meta Learning
- We want the model to adapt to different concepts and tasks, not to different data
- Similar to transfer-learning, but in meta-learning no weight modification is done like in fine-tuning

## What does it mean to train over tasks?
- It mean we from a probability distribution over datasets (not data samples)
- Each task is associated with a dataset

## What is K-shot learning?
- Dataset is a tuple of two subsets
  - Support set S
    - Contains labelled data (N classes)
    - Each sample is actually a tuple of K samples
  - Prediction set B
    - Labelled data that needs to be classified