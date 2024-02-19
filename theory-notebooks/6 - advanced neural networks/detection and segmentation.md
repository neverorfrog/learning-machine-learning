# Computer Vision Problems in Deep Learning

## Semantic Segmentation

- What?
  - Assign a label to each pixel according to the object that it belongs to
  - Operates at pixel level
  - Each pixel is labeled with a semantic category
- How (failing ideas)?
  - Sliding window to classify center pixel
    - Inefficient
  - CNN classification and segmentation on reduced image
    - Impossbile to recover original image size
  - CNN without downsampling
    - Too expensive
- How (final solution)?
  - CNN with downsampling and upsampling phase
  - How to do upsampling?
    - 2x2 patch becomes 4x4 patch
    - Nearest neighbor
    - Max unpooling (saving max from max-pooling)
    - Learnable upsampling (transposed convolution)
- Example: U-Net
  - Contraction phase (downsampling): increases field of view to recover features on the content
  - Expansion phase (upsampling): concatenates feature maps, regenerating high resolution map, recovering position information of features

## Object Detection

- What?
  - Localize and classify multiple objects within the image
  - Operates at object level
- For a single object, object detection
  - Problem can be split into classification and localization
    - Classification with class scores as outputs from FC at end of CNN
    - Regression with bounding box coordinates as outputs from FC at end of CNN
- For multiple objects, how?
  - Slow-RCNN (inefficient)
    - Search of ROIs (regions of interest)
    - Apply CNN on every ROI
    - Classify with SVM
  - Fast-RCNN
    - Apply a CNN on the whole image
    - Extracting ROIs with proposal method from features outputted by CNN
    - Cropping features from ROI
    - Every ROI is fed into a single object detection pipeline (classification and localization)
  - Faster-RCNN
    - ROIs are proposed by a network

## Instance Segmentation

- What?
  - Detect multiple objects and assign a label to every pixel into them
- How?
  - Using Fast-RCNN together with binary mask prediction on the ROI
