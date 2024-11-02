# Camera Calibration

- Goal: estimating the parameters of the projection from a 3D point in the world frame to a 2D in the image frame

## Ways of expressing a point

- We can express a point in 3 ways:
  - World frame: $X_w=(X_w,Y_w,Z_w,1)^T$
  - Camera frame: $X_c = (X_c,Y_c,Z_c,1)^T=[R|t]X_w$
    - $R$ and $t$ are **extrinsic parameters** (rotation and translation)
  - Image frame: $x_I=(x,y,1)=K[R|t]X_w$
    - $K$ (camera matrix) contains **inrinsic parameters** $\alpha_x$ and $\alpha_y$ (focal lengths), $x_0$ and $y_0$ (camera center), $s$ (skew parameter)
- $P=K[R|t]$ is the projection matrix

## Steps for projecting a 3D world point to a 2D image point

- TODO

## Camera Calibration Steps

- Starting point:
  - $
  \begin{pmatrix}
    x_i \\
    y_i \\
    1 \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    P_{11} P_{12} P_{13} P_{14} \\
    P_{21} P_{22} P_{23} P_{24} \\
    P_{31} P_{32} P_{33} P_{34} \\
  \end{pmatrix}
  \begin{pmatrix}
    X_i \\
    Y_i \\
    Z_i \\
    1 \\
  \end{pmatrix}
  $
- We assume that $P_{34}=1$
  - The goal is to find the other 11 elements of $P$

### Linear Method 1

- We rewrite the above equation such that $x_i$ and $y_i$ are known coefficients in vector $b\in \R^{2N}$ where $N$ is the number of points
- We devise a linear system $Ap=b$ where for each point there are two linear equations in 11 unknowns $p$
  - $A\in \R^{2N\times 11}$, $p\in \R^{11}$
- We solve linearly by minimizing energy error function $||Ap-b||^2$ with least squares
  - Depends heavily on invertibility of $A^TA$

### Linear Method 2

- $x_i$ and $y_i$ are not known coefficients anymore, but make $p_{34}$ contribute to the linear system
- We devise a linear system $Ap=0$ where for each point there are two linear equations in 12 unknowns $p$
  - $A\in \R^{2N\times 12}$, $p\in \R^{12}$
- Optimization problem defined with error function $E=||Ap||^2-\lambda(||p||^2 -1)$ that has to be minimized
- Taking derivative wrt $p$ and $\lambda$ we get that $A^TA\hat p=\lambda \hat p$ ($\hat p$ is an eigenvector for $A^TA$) and that $\lambda$ has to be minimized in order to minimize $||Ap||^2$
- Thus, $\hat p$ will be the eigenvector corresponding to the smallest eigenvalue of $A^TA$

### Decomposition of Camera Matrix

- Find camera center with SVD
- Find intrinsic parameters $K$ and rotation $R$ with RQ decomposition
