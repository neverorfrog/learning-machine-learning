# Feature Description

## SIFT Descriptor

- SIFT stays for scale-invariant feature transform
  - Why do we need to be scale-invariant? Because in the same image there could be features with vastly different scales
  - So we need to detect features at every scale
  - Key idea: identify key location in scale-space, such that selected feature vectors are invariant to scaling, rotation or other transformations
- What does it actually do?
  - It is a detector and descriptor in one package
  - It extracts and describes distinctive features in images
    - What does distinctive mean?
      - Features that can be identified across images and invariably to transformations
      - Features that are unique
      - Featues that are robust to noise
  - What do we use it for?
    - Object recognition, image matching, 3D reconstruction, visual SLAM
- Can be summarized in 5 steps

### Step 0: Octave representation

- Generate a multi-scale representation organized in octaves and apply difference of gaussian between each level within an octave
  - One octave contains a fixed set of levels of the gaussian pyramid, where each is obtained by blurring with a gaussian filter with increasing standard deviation by factor k
  - From one octave to the other the image is halved
  - The DoG pyramid highlights regions in the image where there are significant changes in intensity across scales

### Step 1: Multi-scale extrema detection

- Compare pixel with neighbors aside, above and below in scale. It is an extrema, it is a potential keypoint.

### Step 2: Keypoint localization

- To effectively localize keypoints, we need to remove less intense pixels, based on a treshold

### Step 3: Orientation assignment

- Why? Provides rotation invariance
- We divide the image in patches and compute a histogram of gradients for each patch
  - We compute gradient orientation and magnitude for each patch
  - HoG has orientation bins on the x-axis (360 deg divided eg by 10) and sum of weighted magnitudes for each bin
- Every bin with value above 80% is a keypoint with that orientation

### Step 4: Keypoint description

- Once we assigned position, scale and orientation, we need to describe the keypoints such that we are invariant to these image transformations
- Steps:
  - Use corresponding scaled image to keypoint
  - Image gradient in 16x16 patch around keypoint
  - Rotate the gradient according to keypoint orientation
  - Divide patch in 4x4 cells
  - Compute orientation histogram for each cell with 8 orientation bins
  - This results ina 128 long vector

### Step 5: Descriptor Matching

- We need a distance function between descriptors

## HOG Descriptor

- Goal is to describe image patches while being invariant to scale, rotation ecc
- Steps
  - Compute x and y gradient for each pixel
  - Compute gradient histogram for each 8x8 cell
    - We divide the orientation into bins, and for each orientation in the gradient of the patch, the magnitude is added to the corresponding bin
    - Add the contributions of all pixels in the cell to create a 9-bin histogram
  - Unite adjacent cells to form biggers cells and normalize to be robust to lighting variations
