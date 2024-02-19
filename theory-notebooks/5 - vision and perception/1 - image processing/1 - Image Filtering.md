# Image Filtering

**What is a grayscale image?**

- It is a function $f: 2D \rightarrow \R$
- $f(x,y)$ indicates the pixel intensity in that position
- A digital image is a **discrete** version of that function
- **A colored image is just a stack of three of that functions**

**What exactly does it mean to discretize an image?**

- It means sampling and quantizing (rounding to nearest neightbor)
- The samples will be $\delta$-distant

**Types of image transformations:**

- Filtering: changes pixel values, forming a new image whose pixel values are a linear combination of the original
- Warping: changes pixel positions

## Filtering

- What do we need it for?
  - To extract features (corners, edges)
  - To enhance images (denoising, deblurring)

### Point Processing

- Each pixel is treated independently
- New image result from applying the same operation to each pixel
- Practical applications?
  - Lighten, darken, invert, contrast ecc.

### Linear shift-invariant Filtering

- What are we doing intuitively?
  - Replace each pixel by a linear combination of its neighbors
  - Each linear combination determined by the **kernel**
- We can express it mathematically as a convolution
  - Suppose we have a quadratic kernel $K$ and an image $I$
  - Convolution: $(K*I)(x,y)=\Sigma_{i,j} K(j,i) \cdot I(x-j,y-i)$
  - If the kernel is of size $(k,k)$, $i$ and $j$ will go from $-k$ to $k$ in the sum
  - In practice we flip the kernel and slide it over the image
- What are the properties of convolution?
  - It is a multiplication-like operation, so commutative, associative

#### Linear Separable Filters

- What does it mean if a filter is linearly separable?
  - The filter can be expressed as an outer product of two 1-D vectors
- Why do we need filters to be linearly separable?
  - Because convolution is computationally more efficient with them
  - It's like doing two 1-D convolutions (cost is $2*K*M*N$ where $(M,N)$ is the image size)

#### Example: Box Filter

- Constant matrix equal to 1/R if R is the number of elements in the filter
- What's its effect?
  - Blurring, because every pixel takes the mean among the surrounding pixels

#### Example: Gaussian Filter

- Filter values are sampled from a gaussian distribution
- So weights fall off with distance from center
