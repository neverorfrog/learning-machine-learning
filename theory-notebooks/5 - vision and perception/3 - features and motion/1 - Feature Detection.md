# Features

- We should find local features that are invariant to transformation
- How do we handle these features?
  - Feature detection: identify interest points that are likely to match well in other images
  - Feature description: extract invariant and compact description from around interest point
  - Feature matching: correspondende between features from different views
  - (Feature tracking if there was movement)
- Detecting unique features is important for many tasks
  - visual slam, 3D reconstruction, image matching and stitching etc...
- But how do we measure the uniqueness degree of a feature?
  - With corners

## Harris Corner Detector

- Intuition
  - A corner happens when there is a change in pixel intensity in multiple directions
  - We slide a small window over the image by small variations in position. But how do we measure intensity change in direction $[u,v]$?
    - Expressed as a bilinear form $E(u,v)=[u,v]^TM[u,v]$
    - But how do we get to this form?
      - We take the error function $E(u,v) = \Sigma_{x,y}w(x,y) [I(x+u,y+v)-I(x,y)]^2$ and approximate it at first order
      - So the locally the intensity change is a surface: corners correspond to holes, edges to archs
- Steps:
  - Compute x and y gradients of image and center and centering them through the mean
  - Compute the covariance matrix M at each pixel (defined as? TODO) that gives a measure of the intensity change
  - Compute the response to the detector by defining the threshold $R = \lambda_1 \lambda_2 - k(\lambda_1+\lambda_2)^2$
    - $\lambda_1$ and $\lambda_2$ are the eigenvalues of $M$, which can be seen as the axes of the ellipse that comes out if we put $E$ to a constant value
    - For each pixel where $R >> 0$ there is a corner
  - Where a corner is identified we do non-max suppression in the neighborhood

## Scale-Invariant Detection

- A good feature detector should be invariant to scale, rotation, lighting etc...
- Harris corner detection is not invariant to scale
- How do we build one that is? How can we automaitcally select scale?
  - We need to detect blobs with an appropriate filter and detect local maxima
  - So, we design a **function** on the region (circle), which is “scale invariant”
(the same for corresponding regions, even if they are at different scales)
  - Then, find scale that gives local maximum of f in both position and scale
- Steps:
  - Scale-Space Representation: Gaussian Pyramid: scale-space representation of the image, where details at different scales are preserved.
  - Blob Detection: At each scale level, potential blob candidates are identified by detecting regions where the intensity or color significantly differs from the surrounding regions. LoG.
  - Blob Localization: Once potential blob candidates are identified, they are localized more precisely by finding the local maxima/minima in the scale-space representation. These maxima/minima correspond to the centers of the detected blobs.
  - Scale Selection: Blobs detected at different scales are often filtered based on certain criteria (e.g., blob size, response strength) to select the most relevant blobs.
