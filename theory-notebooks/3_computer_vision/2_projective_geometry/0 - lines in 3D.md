# Lines in 3D

- We can express a line in 3D as $l=p_0+d_0s_0$
  - $p_0 \in \R^{3}$ is an offset
  - $d_0 \in \R^{3}$ expresses a direction
  - $s_0 \in \R$ is a scalar ascissa

## Intersection Condition

- Two lines intersect if
  - Given direction $d_0 \times d_1$ orthogonal to both lines
  - Given difference $p_0 - p_1$ between two points on the two lines
  - It holds that $(p_0 - p_1) \cdot (d_0 \times d_1)=\bold 0$
- Intuitively, if the projection of the difference vector between two points along the orthogonal direction to both lines is zero
  - Which means the difference vector and the orthognal direction are aligned

## Problem: computing closest point between two non-intersecting lines

- Solution template: find the orthogonal direction to both lines and find the intersection of this orthogonal line to the two input lines
- How?
  - Definining a distance between two points $\Delta(s_0,s_1)$
  - Minimizing the squared norm of $\Delta$
  - Solution is the pseudo-inverse