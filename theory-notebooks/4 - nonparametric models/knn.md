# K Nearest Neightbors

- Instance based learning method
  - Do not use any fixed number of parameters
  - Instead the parameters grow with the amount of data used
  - In fact, in KNN the whole dataset is used to "learn" a function

- Algorithm Idea
  - Classification problem with target function $f: X \rightarrow \{C_1,...,C_I\}$
  - The training phase consists simply in saving the whole dataset
  - Then, we compute the posterior $P(C_i|x,D,K)=\frac{1}{N}\Sigma_{x_n \in N_K(x_n,D)} (t_n == C_i)$, where $N_K$ is the set of the K nearest neighbors according to a certain distance function
    - Intuitively, we take a vote among the k nearest neighbors and assign the most frequent label to the to-be-classified sample
- What about the distance function?
  - Algorithm depends heavily on the distance function
  - We can use any distance function than just euclidean distance through the kernel substitution

- And linear regression? Can KNN be used also for that?
  - Yes, we basically take still k neirest neighbors
  - And then fit a local linear regression model

That's it
