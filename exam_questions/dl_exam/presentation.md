# Task 

We will first give some insights about the task. We are in the context of binary classification between single and bifurcating arteries, given a dataset of meshes, from which some features are already extracted.

On the top figures we can already see that the dataset is almost linearly separable. The only problematic feature is the position.

Also, on the bottom left figure we can observe that the dataset is very balanced between the two labels.

On the bottom right table, instead, we report the shapes of a single datapoint. As we can see, their shapes are not all the same, and for this reason we chose to down-sample every sample to 300 items. This was done also for computational reasons. Although, we did not notice any performance drop with respect to increasing the number of items after the down-sampling. Every sample has size [num_items, vector_size]. Each item will be embedded into the geometric algebra framework as we will see in a few slides.

# Motivations

### Why Geometric Algebra?

The main reasons to adopt geometric algebra for this kind of task is because geometric algebra allows us to define an inductive bias for geometric data, namely points, planes, but also translations and rotations. The goal is have an architecture that respects symmetries in 3D space, and therefore the architecture should be equivariant to translations, rotations and reflections in E(3). The advantage of geometric algebra is that it also comprises translations, so it offers a complete framework for all the geometric operations needed for this task.

### Why Transformers?

Because it's easy to train, scalable and adaptable to different data types.

# Equivariance

- Applying a transformation ρ to the input x and then passing it through the layer f must yield the same result as first applying f to x and then applying the transformation ρ to the output.
- To verify the equivariance of the proposed layers, we leveraged a property of orthogonal transformations: they can be expressed as sequences of reflections.
  - Possible to verify the equivariance of any group action by
verifying the equivariance with respect to sequences of reflections. 
  - Specifically, we compute a versor u as a sequence of random reflections
- So in this particular case $\rho$ is actually the sandwich product
- The results show that all layers passed the equivariance test except the
Attention layer. This is most likely due to an extension proposed in the
paper, which suggests using a distance-aware dot product to make the
attention layer fully equivariant

# Results

- Here we report the metrics blablabla
- As we can see the gatr has the minimum loss 
- The loss also converges more quickly
