# Fourier Transform

[3Blue1Brown](https://www.3blue1brown.com/lessons/fourier-transforms)

## Goal

- If we have a sum of sinusoids we want to decompose into the single sinusoids (one for each frequency)

## Sinusoid is an oscillating and rotating vector on unit circle

- Euler formula: $x + iy = cos \rho + i sin \rho = e^{i\rho}$
  - $\rho$ is how much, in radians, i am far away from 0 going counterclockwise
  - $e^{i2\pi}$ is a full rotation
- Rotation in function of time: $e^{i2\pi t f}$
  - $t$ is time in seconds
  - $f$ is the (wound-up) frequency: how much rotation is done every second
- Finally, if my sinusoid is $g(t)$, the wound-up version is $g(t) \cdot e^{i2\pi t f}$
  - We have two frequencies
    - Wound-up frequency $f$
    - Frequency of the sinusoid $g(t)$

## Definition of Fourier Transform

- Wound-up graph have a center of mass
  - We sample many points on $g(t)$ and make an average
- So FT is integral that gives complex number from a sinusoid in time
  - $\hat g(f) = \int{g(t) \cdot e^{-i2\pi t f} dt}$
- This number represents the strength of the sinusoid in that frequency
  - Intuitively it is the distance of the com of the wound-up graph from the origin

## Discrete Fourier Transform

- Signal s of length $N$
- $\zeta = e^{-\frac{2\pi i}{N}}$ is one nth of a rotation around the unit circle
  - it is a vector in the complex plane
- $\zeta ^ f$ means we are rotating around unit circle at frequency $f$
  - in the same time the signal ends, we have rotated $f$ times around the circle
- $\hat s[f] = \sum_{n=0}^{N-1} s[n] \cdot \zeta^{f n}$
  - given f, we multiply each sample of s (a scalar in 1D) with the nth vector if we would unitcircling at that frequency
- practically a weighted mean of the vectors at a certain frequency, with weights the samples of the signals
- if the weighted mean (com) is far away from the origin, that frequency is present in the signal
