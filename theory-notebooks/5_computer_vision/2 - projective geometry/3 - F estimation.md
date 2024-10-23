# Estimating $F$

- How many correspondences $x \leftrightarrow x'$ do we need to estimate F?
  - 7 because F is of 7dof
- How can we devise a linear system for estimating F?
  - We flatten by rows $x'x^T$, generating a 9-dim vector $A_i^T$
    - Will be a row of matrix $A\in \R^{N\times 9}$
    - $N$ is the number of point correspondences
  - We flatten $F$ by rows, generating a 9-dim vector $f$
  - The system is then $Af=0$
- How do we solve this system?
  - $rank(A)<8 \rightarrow$ overdetermined
    - Exact solution exists (up to scale) but is not unique
  - $rank(A)=8$
    - Exact solution exists (up to scale) and is unique
  - $rank(A)=9 \leftarrow$ data is noisy
    - We have to adopt least-squares: 8-points

## 8-points algorithm

- Input
  - n-point correspondences
  - But how are the correspondences found?
    - RANSAC (founds them without outliers)
    - Normalization between [-1,1] or [0,1]
- Output
  - Estimated fundamental matrix $F'$
- Steps
  - Construct homogeneus system $Af=0$ from $x'^TAx=0$
  - First estimate $\hat F$ corresponds to the smallest singular value of $A=UDV^T$
  - Impose singularity constraint since $rank(F)=2$
    - Set smallest singular value of $UDV^T$ to 0, obtaining $D'$
  - Obtain final estimate $F'=UD'V^T$