"""Radix-2 Cooley-Tukey FFT over complex numbers.

The public functions accept and return lists of complex numbers. The FFT is
iterative and requires the input length to be a power of two; callers that
need other lengths should zero-pad before calling. That restriction is the
only trade-off: it keeps the implementation small and predictable without
resorting to Bluestein's algorithm.
"""

from __future__ import annotations

from cmath import exp, pi
from typing import List


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _bit_reverse_indices(n: int) -> List[int]:
    """Return the bit-reversal permutation for indices in [0, n).

    The FFT is easiest to write iteratively if we first reorder the input into
    bit-reversed order. We compute the permutation once and reuse it for both
    forward and inverse transforms.
    """
    bits = n.bit_length() - 1
    result = [0] * n
    for i in range(n):
        rev = 0
        x = i
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        result[i] = rev
    return result


def _fft_in_place(values: List[complex], inverse: bool) -> None:
    n = len(values)
    if not _is_power_of_two(n):
        raise ValueError("FFT length must be a power of two")

    rev = _bit_reverse_indices(n)
    for i in range(n):
        j = rev[i]
        if i < j:
            values[i], values[j] = values[j], values[i]

    length = 2
    while length <= n:
        # The twiddle factor is exp(-2*pi*i/length) for the forward transform
        # and its conjugate for the inverse. Using one formula avoids a second
        # table and keeps the sign handling in one place.
        angle = (2.0 * pi / length) * (1.0 if inverse else -1.0)
        wlen = exp(1j * angle)
        half = length // 2
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = values[start + k]
                v = values[start + k + half] * w
                values[start + k] = u + v
                values[start + k + half] = u - v
                w *= wlen
        length <<= 1

    if inverse:
        for i in range(n):
            values[i] /= n


def fft(values: List[complex]) -> List[complex]:
    """Compute the discrete Fourier transform of ``values``.

    ``values`` must have a length that is a power of two. The result is a new
    list; the input is not modified. The forward transform uses the sign
    convention exp(-2*pi*i*k*n/N).
    """
    data = list(values)
    _fft_in_place(data, inverse=False)
    return data


def ifft(values: List[complex]) -> List[complex]:
    """Compute the inverse discrete Fourier transform of ``values``.

    ``values`` must have a length that is a power of two. The result is a new
    list; the input is not modified. This is the inverse of :func:`fft` for
    inputs produced by it, up to floating-point rounding.
    """
    data = list(values)
    _fft_in_place(data, inverse=True)
    return data


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def convolve(a: List[complex], b: List[complex]) -> List[complex]:
    """Linear convolution of two sequences via FFT.

    The result has length ``len(a) + len(b) - 1``. Both inputs are zero-padded
    to the next power of two at least that long, so the circular convolution
    computed by the FFT matches the linear convolution. An empty input yields
    an empty result.
    """
    if not a or not b:
        return []
    result_len = len(a) + len(b) - 1
    n = _next_power_of_two(result_len)
    fa = fft(list(a) + [0j] * (n - len(a)))
    fb = fft(list(b) + [0j] * (n - len(b)))
    product = [x * y for x, y in zip(fa, fb)]
    return ifft(product)[:result_len]


def multiply_polynomials(a: List[complex], b: List[complex]) -> List[complex]:
    """Multiply two polynomials given by coefficient lists.

    Coefficients are ordered from the constant term upward. This is exactly
    the linear convolution of the coefficient sequences, so it delegates to
    :func:`convolve`. An empty coefficient list represents the zero polynomial
    and yields an empty result.
    """
    return convolve(a, b)
