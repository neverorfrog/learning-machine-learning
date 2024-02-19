# Frequency and Image Pyramids

- What does frequency represent for images?
  - Spatial frequency in a digital image refers to how rapidly pixel intensities change in the spatial domain, i.e., across the image's rows and columns. High spatial frequency indicates rapid changes in pixel values, such as edges, textures, or patterns, while low spatial frequency represents gradual changes or smooth regions.

## Fourier Transform

- Is a way of analizing images in the frequency domain
- Yields a transformation where for each frequency couple of the image there is some intensity
- A convolution in the frequency domain is just a product $\rightarrow$ better analysis of linear filters
- Frequency domain allows to downsample or upsample images in the right way $\rightarrow$ image pyramids

## Image Pyramids

- What if we want to change the resolution of an image?
- For example we want to resolve an aliasing problem
  - What is the Nyquist frequency?:
    - It is the maximum frequency that can be faithfully captured or represented in a digital signal sampled at a particular rate
    - In practice the half of the sampling frequency
    - It means that if an image has $f_{max}$ as highest frequency, necessarily $f_s \geq f_{max}$
  - Aliasing happens when the image is sampled at a too low frequency
  - To solve this, we could raise the sampling frequency (oversample) or smooth the image (downsample)
- What are these image pyramids used for?
  - To compress images
  - Or also to blend images
  - Or also to denoise images
  - Or also to do multi-scale analyisis

### Gaussian Image Pyramid

- We reduce the resolution of the image progressively, by also smoothing it
  - That means reducing the frequency
- We construct a pyramid made of multiple levels
  - Each time we go a level down, the image is smoothed and the pixels are reduce
  - That until we reach minimum resolution
  - So practically details get blurred out while we go up the pyramid

### Laplacian Image Pyramid

- In addition to smoothing and subsampling, between those two steps we retain the residuals and not the entire image itself
- What are residuals?
  - Difference between the original and the processed image
  - Difference of gaussians
  - Approximates the Laplacian
- This way the original image can be reconstructed from the pyramid

## Linear Filters in the Frequency Domain

- High-pass, to detect edges
- Low-pass, to blur
  - Gaussian blur is a low-pass filter
  - Lowers the frequency
