# Distance Learning

## What if I wanna cluster the data? METRIC LEARNING (NOT A SSL TASK)

- Learning a distance in sample space
- We can use the triplet loss
  - $\mathcal{L}(x_a,x_+,x_-) = max(0, \mathcal{D}_{f_{\theta}}^2(x_a, x_+) + \epsilon - \mathcal{D}_{f_{\theta}}^2(x_a, x_-))$
  - This loss ideally has to go to zero
  - If it is negative, it means the distance to the negative sample is bigger than the distance to the positive sample
  - Minimizing this loss means making $\mathcal{D}_{f_{\theta}}^2(x_a, x_+)$ small and $\mathcal{D}_{f_{\theta}}^2(x_a, x_-)$ big

### Problem: the blobs are stretched and not so far apart

- We can mitigate this by adding a regularization term that enforces the loss to reduce the distance from the center of the centroid
- This way the shape is more circly