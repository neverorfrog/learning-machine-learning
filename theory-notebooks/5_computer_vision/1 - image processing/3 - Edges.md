# Edges

- Defined as a discontinuity in the image
  - Due to depth, color, illumination ecc...
  - Practically, it's a sudden change in the intensity function, or otherwise an extremum of the derivative
- Why are edges important?
  - More compact representation than pixels
  - Semantic and shape information can be easily obtained
- **What happens when we reach the edge of the image?** $\rightarrow$ **Padding**
- But first, **how do we detected edges?**
  - Intuitively, by taking derivatives (because they are large at discontinuities)
  - We could find edges by simply inspecting pixel intensities of the gradient, **but** noise breaks the balls
  - To solve this we blur before derivating $\rightarrow$ canny-edge detector, or derivate the blur

## Canny-Edge Detector

- Goal is to find edges without false positives or false negatives and pinpointing edges where they actually occur
- Steps:  
  1. Filter image with derivative of the gaussian filter (or laplacian of gaussian to have higher performance)
  2. Compute magnitude and direction of gradient
  3. Non-maximum suppression, aka we eliminate pixels that may not constitute the edge by checking for every pixel if it is the maximum in its neighborhood
  4. Linking and thresholding (hysteresis), where we connect strong pixels to non-weak pixels (if they are in fact connected)
