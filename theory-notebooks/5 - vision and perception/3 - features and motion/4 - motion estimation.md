# Motion Estimation

- What?
  - Estimating pixel transformations from one image to another inside a video
  - Thus, our image data is a function of space $(x,y)$ and time $t$
  - Motion of pixels? Caused by movement of a scene or the camera
- Why?
  - Object tracking, super-resolution, video compression, event recognition, segmentation based on motion cues
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

---
