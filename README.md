# fast-fourier-transform-py

A small standard-library-only implementation of the radix-2 Cooley-Tukey fast Fourier transform over complex numbers, plus helpers for linear convolution and polynomial multiplication.

```python
from fast_fourier_transform_py import fft, ifft, convolve, multiply_polynomials

spectrum = fft([1, 2, 3, 4])
signal = ifft(spectrum)

product = convolve([1, 2, 3], [0, 1, 0.5])
# polynomial (1 + x)(1 - x) = 1 - x^2
poly = multiply_polynomials([1, 1], [1, -1])
```

## Why this exists

Python already has numerical FFTs, but they live in packages that are not part of the standard library. When the environment has no network access and no third-party packages, a compact dependency-free transform is useful for exact-length workloads such as convolving short coefficient sequences. The deliberate trade-off is that the transform length must be a power of two: the implementation rejects other lengths with `ValueError` instead of silently falling back to a slower mixed-radix or Bluestein algorithm. That keeps the code small and the failure mode obvious.

## API

- `fft(values)` — forward transform. `values` is a list of complex numbers whose length is a power of two. Returns a new list.
- `ifft(values)` — inverse transform. Same length restriction. Returns a new list.
- `convolve(a, b)` — linear convolution of two sequences, using zero-padding to the next power of two at least `len(a) + len(b) - 1`. Returns an empty list if either input is empty.
- `multiply_polynomials(a, b)` — polynomial multiplication with coefficients ordered from the constant term upward. Delegates to `convolve`.

## The awkward edge

The power-of-two restriction is the thing you will hit first. A call such as `fft([1, 2, 3])` raises `ValueError`; it does not pad for you. `convolve` and `multiply_polynomials` pad internally, so they accept arbitrary lengths. If you need a transform of a different length, zero-pad your input to the next power of two yourself and remember that the output then has that padded length.
