# Epipolar Geometry

- It's the geometry of stereo vision
- What?
  - **Correspondance geometry**: given two image points $x, x'$ from two different 3D views, what are the geometric constraints between them?
  - **Camera geometry (motion)**: given two image points $x_i, x_i'$ what are the cameras $P, P'$ for the two views?
  - **Scene geometry (structure)**: given two image points $x_i, x_i'$ and two cameras $P,P'$, what is the position of the the captured 3D point $X$?
- How? Exploiting the camera geometry (approximated to pin-hole camera model) and scene structure
- For what?
  - Depth estimation
  - 3D reconstruction
  - Structure in motion

## Antipasto: Lines in 3D

- We can express a line in 3D as $l=p_0+d_0s_0$
  - $p_0 \in \R^{3}$ is an offset
  - $d_0 \in \R^{3}$ expresses a direction
  - $s_0 \in \R$ is a scalar ascissa

### Intersection Condition

- Two lines intersect if
  - Given direction $d_0 \times d_1$ orthogonal to both lines
  - Given difference $p_0 - p_1$ between two points on the two lines
  - It holds that $(p_0 - p_1) \cdot (d_0 \times d_1)=\bold 0$
- Intuitively, if the projection of the difference vector between two points along the orthogonal direction to both lines is zero
  - Which means the difference vector and the orthognal direction are aligned

### Problem: computing closest point between two non-intersecting lines

- Solution template: find the orthogonal direction to both lines and find the intersection of this orthogonal line to the two input lines
- How?
  - Definining a distance between two points $\Delta(s_0,s_1)$
  - Minimizing the squared norm of $\Delta$
  - Solution is the pseudo-inverse

## Back to Epipolar Geometry: Ingredients

- 1 3D point $X$
- 2 Camera centers $C, C'$
- 2 2D image points $x,x'$
- 2 image planes

## Definitions

### Epipolar Plane $\pi$

- Plane connecting the two camera centers $C, C'$ and $X$
  - Line connecting $C-C'$ is the **baseline**
- $x, x'$ lie on that plane, respectively on the lines $X-C$ and $X-C'$

### Epipolar Lines $l,l'$

- Intersection of $\pi$ with the image planes
- $l'$ corresponds to projection of point $x$ to the other image

### Epipoles $e,e'$

- Intersection point between baseline and the image planes
- They also lie on $l,l'$
- Projection of $C$ onto the plane of $C'$

### Epipolar pencil

- Generated if X is shifted around
- Family of epipolar planes
  - All epipolar planes contain the baseline
- All epipolar lines intersect in the epipoles $e,e'$
  - That's why the epipoles are also the vanishing points of camera motion direction

## Wrapping up: Fundamental Matrix $F$

- Epipolar constraint between $x$ and $x'$, namely that $x'$ must lie on $l'$
- $l'=Fx$
  - F is a 3x3 projective mapping, not invertible
  - 7 dof because of rank 2
- Since $x'$ lies on $l'$ we rewrite that $x^Tl'=0$
  - Thus $x^TFx=0$
- How is it obtained?

### Geometric Derivation

- $l'=e' \times x'$
  - line through space expressed by the vector orthgonal to it
- $x' = H_{\pi} x$
  - Homography mapping through epipolar plane
- $l'=e' \times H_{\pi} x = F x$

### Algebraic Derivation

- TODO

### 8-Point Algorithm

## Essential Matrix
