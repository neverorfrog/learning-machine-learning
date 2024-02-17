# Image Alignment

- Suppose we matched two features A and B from different images
- Aligning means finding the transformation between feature A and B
  - A transformation is identified by a matrix
  - So we need to find the parameters of that matrix
- We can define the residual $r_{x_i}=x_i+x_t-x_i'$ as the error between transformation of feature in this image and actual feature in other image
- If we have many features, the idea is to do least squares to minimize squared sum of residuals, so linear regression
- We need to maximise inliers (points that agree very much with the line). That's where RANSAC comes into play

## RANSAC

- Basic idea: all the inliers agree with each other, while the outliers don't
- Iteratively for N times
  - Choose randomly s samples (image feature pairs) and fit a model (eg a line)
  - Count the inliers
- Choose model with most inliers

## Image Stitching

- Create panoramas, namely combining two images together
- We need to compute a homography using RANSAC
  - Homography is an affine transformation + projective warp
  - TODO
