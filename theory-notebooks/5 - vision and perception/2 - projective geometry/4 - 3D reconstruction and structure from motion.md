# 3D Reconstruction

- Problem
  - Given $F$ or the $P,P'$, find the 3D point $X$ that maps to corresponding points $x,x'$
  - Back-projected ray do not intersect cause of estimation errors
  - Reconstruction ambiguity
    - reconstruction is defined up to a projective transformation
- Solution idea
  - Triangulation (not projective/affine invariant)
    - We solve $Ax=0$
    - Homogeneus, discrete for all cameras
    - Inhomogeneus, good for calibrated cameras
  - Geometric error
    - Minimize distance in the image space

## Structure from Motion

- Input
  - $m$ images of $n$ fixed 3D points $X_j$
  - $m*n$ correspondences $x_{ij} \leftrightarrow x_{kj}$
    - detected with SIFT and matched with RANSAC
- Output
  - $m$ projection matrices
  - $n$ points $X_j$
- Constraints
  - Projective/affine ambiguity
- Solution
  - Sequential structure from motion
    - Determine projection matrix with 2D-3D correspondence
    - Iteratively refine and extend the 3D structure by triangulation
  - Bundle adjustment
    - Use non-linear method by minimizing reprojection errors