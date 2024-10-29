# Motion Estimation

- What?
  - Estimating pixel transformations from one image to another inside a video
  - Thus, our image data is a function of space $(x,y)$ and time $t$
  - Motion of pixels? Caused by movement of a scene or the camera
- Why?
  - Object tracking, super-resolution, video compression, event recognition, segmentation based on motion cues, visual odometry
- How?
  - Optical Flow, recover image motion at pixel level

## Optical Flow

- What?
  - Estimation of **apparent** motion of brightness patterns in the image
    - Apparent because cause by lighting changes
  - Can be **sparse or dense**
- How?
  - We want to estimate the **motion field**
    - Projection of 3D scene velocities onto the image plane
    - Motion between two frames happens in $\delta t$ with velocity $u$ along $x$ and $v$ along $y$
    - Spatial shift definable as $\delta_x$ and $\delta_y$
  - Optical flow has to estimate $u,v$
  - *But* we need to do some approximations in order to make the optical flow equivalent to the motion field
    - **Brightness constancy**: brightness doesn't change
    - **Small motion**: points don't move much
    - **Spatial coherence**: points move like their neighbors
  - *Thus* we will look for nearby pixels with the same color
    - From assumptions the brightness is: $I(x+\delta_x,y+\delta_y,t+\delta t)=I(x,y,t)$
    - Assuming small motion and expanding with Taylor we get the **brightness constancy equation** $I_xu+I_yv+I_t=0$
- **Aperture Problem**
  - We cannot estimate both $u$ and $v$ because we have just 1 constraint
  - Locally, we can only estimate the normal flow $g_n$
    - $g=[u,v]=g_n+g_p$ is the optical flow
    - $g$ is constrained to live on the line expressed by brightness constancy equation
    - we can compute distance from origin to this line (namely $g_n$)
  - This can be solved in 2 ways

---

### Lucas-Kanade Optical Flow

- Assumptions?
  - We can consider image per patches (method is sparse)
  - Flow is constant for a neighborhood patch
  - Neighborhood pixels have the same displacement
- Result
  - System of $N^2$ equations in two unknowns $u,v$ where $N$ is the number of pixels in the patch
  - $A \cdot [u,v]=b$
    - $A \in \R^{N^2 \times 2}$ where $A_i^T=[I_x(p_i), I_y(p_i)]^T$
    - $b \in \R^{N^2}$ where $b_i=-I_t(p_i)$
    - Can be solved with least squares
      - $[u,v] = A^+b$
      - Relies on big enough eigenvalues for invertibility of $A^TA$
      - For edges can estimate motion only if it is normal enough to the edge
- **Important**
  - $A^TA$ is the second moment matrix of the Harris detector
  - Largest eigenvalue points in the direction of fastest change
  - Eigenvalues both large: corner
    - Correspond to high texture regions where lukas-kanade works best
    - Because if we have visibly different gradients we avoid the aperture problem as much as possible

### Horn-Schunck Optical Flow

- Assumptions?
  - Objects move with smooth lines
  - Flow field is smooth
- Result?
  - Global energy-like function $E=\Sigma_{i,j}[E_s+\lambda E_b]$
    - Contains smoothness and brightness constancy
    - $E_s(i,j)=\frac{1}{4} [(u_{ij}-u_{i+1,j})^2 + (v_{ij}-v_{i+1,j})^2+(u_{ij}-u_{i,j+1})^2+(v_{ij}-v_{i,j+1})^2]$
    - $E_b(i,j)=[I_xu_{ij}+I_yv_{ij}+I_t]^2$
  - If minimized, gives the flow field
    - Solved iteratively

---

## What if the small motion assumption is not valid?

- Problem
  - If there are bigger movements, for which we can't use first order taylor approximation anymore?
  - The brightness constancy does not hold
- **Solution: Resolution Pyramids**
  - Idea
    - The pixel intensity difference between two frames decreases as the resolution decreases
    - With sufficient low resolution, small motion becomes valid again
  - Procedure
    - We have two successive frames: A and B
    - We compute optical flow between A and B at low resolution
    - Then, we reconstruct B by warping A with the low-resolution optical flow
    - The optical flow we want is then computed between the reconstruced B and the real B

---

## What if the spatial coherence assumption is not valid?

- Problem
  - Objects may be at very different distances from the camera
  - Near objects move faster than far objects
- **Solution: Motion Segmentation**
  - We break the image into layers
  - Each layer has a coherent motion
