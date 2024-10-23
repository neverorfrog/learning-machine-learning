# Some useful definitions

## Patch

Sub-image centered at $(i,j)$ and containing all pixels at distance lower than a certain threshold $k$:  
$P_k(i,j)$

## Invariance

A function is shift invariant if the output does NOT change when the input is shifted:  
$ f(T(x)) = f(x) $

## Equivariance

A function is shift equivariant if the output shifts the same way as the input, when the latter undergoes a shift:  
$ f(T(x)) = T(f(x)) $

## Embedding

Process that transforms data from a non-metric space (images, audios etc...) to a **metric space**, where they are governed by a certain algebra

## Latent Space

- Result of an embedding process
- Typically higher-dimensional vectors when we are extracting features from raw data
- In theory a space where the actual data distribution lives
- We learn the geometry of the sample space