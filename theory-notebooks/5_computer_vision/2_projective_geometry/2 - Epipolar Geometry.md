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

## Ingredients

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

- What?
  - Algebraic representation of epipolar geometry
  - 3x3 matrix with rank 2, not invertible, with 7 dof
- Assumption
  - **Epipolar constraint** between $x$ and its corresponding point $x'$, namely that $x'$ must lie on $l'$
- Result
  - There is a mapping from $x$ to $l'$ through F

### How do we get there?

#### Geometrically (with epipolar lines $l,l'$)

- $l'=e' \times x'=[e']_{\times}x'$
  - line through space expressed by the vector orthgonal to it
  - $[e']_{\times}$ is skew-symmetric (rank 2)
- $x' = H_{\pi} x$
  - Homography mapping through epipolar plane
- $l'=[e']_{\times} H_{\pi} x = F x$
- Since $x'$ lies on $l'$, $x'^Tl'=0$, thus $\mathbf{x'^TFx=0}$
- $F$ represents a mapping from a 2-dim onto a 1-dim projective space, and hence must have rank 2

#### Algebraically (with projection matrices $P,P'$)

- Here we also need an expression for $l'=e' \times x'$
  - $e'=P'C$ is the projection of C onto second image plane
  - $x'=P'X=P'P^+x$, since $x=PX$
- In the end $l'=[e]_{\times}P'P^+x$, thus $F=[e]_{\times}P'P^+$

## $F$ with some motions

### Pure Translation

- $P=K[I|0], P'=K[I|t]$
- $F=[e']_{\times}$
  - Only 2 dof
- $x'=x+\frac{Kt}{Z}$

### General Motion

- TODO

## Projective ambiguity and canonical form

- Problem: $F$ does not depend on the world frame
  - statement 1: map from $P,P'$ to $F$ is not injective
    - we cannot uniquely retrieve $P,P'$ from $F$
  - statement 2: $F$ is invariant to a 3D projective transformation H
    - $F$ from $P,P'$ and $F'$ from $PH,P'H$ are the same
  - statement 3: if there are two couples $P,P'$ and $\tilde P,\tilde P'$, and $F$ relates to both couples, there exists a projective transformation H relating the two couples
- Solution: canonical form and skew-symmetry to the resque
  - statement 1: if $P=[I|0]$ and $P'=[M|m]$, then $F=[m]_{\times}M$
  - statement 2: we can determine $F$ just from a pair $P,P'$ if $P'^TFP$ is skew-symmetric
  - statement 3: we choose $P=[I|0]$ and $P'=[[e']_{\times}F|e']$

## Essential Matrix

- What if the cameras are calibrated?
- We can normalize points in the image plane. But how?
  - $\hat x=K^{-1}x$
- If $P=[I|0]$ and $P'=[R|t]$ are normalized camera projection matrices, then $F=[t]_{\times}R$
- Relation between essential and fundamental?
  - $\hat x'^TE\hat x=0$
  - $x'^T K'^{-T}EK^{-1}x$
  - Thus $F=K'^{-T}EK^{-1}$
