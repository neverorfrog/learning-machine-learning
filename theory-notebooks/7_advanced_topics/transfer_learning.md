# Transfer Learning

- Problem:
  - We have a source task $T_s$ where we train a model $L_s$ on a dataset $D_s$ that learns a function $f_s$
  - Suppose the model $L_s$ is trained and works
  - We also have a target task_ $T_t$ with a different dataset, model and function to learn (eg different image format and distribution, different classes)
  - The goal of transfer learning is to use the learned model $L_s$ for $T_r$, so to reduce computational effort of possibly re-learning stuff we potentially already know from $T_s$

## Fine-Tuning

- Used especially if we have many data
- Consists in freezing a part of the trained layers (mostly the first half or so)
- We just train the remaining layers on the new dataset
- Intuitively, we are retaining some low-level features (edges, corners) which are usually recognized by early layers and training on the higher level features

## CNN as feature extractor

- If the two tasks are similar enough, we may also retain the whole CNN and use it as a feature extractor as is
- On top of the last layer we train a linear classifier like SVM
