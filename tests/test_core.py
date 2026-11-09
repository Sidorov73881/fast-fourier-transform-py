import cmath
import math
import unittest

from fast_fourier_transform_py import fft, ifft, convolve, multiply_polynomials


def assert_complex_close(test, got, want, places=9):
    test.assertEqual(len(got), len(want))
    for g, w in zip(got, want):
        test.assertAlmostEqual(g.real, w.real, places=places)
        test.assertAlmostEqual(g.imag, w.imag, places=places)


class TestFFT(unittest.TestCase):
    def assert_complex_close(self, got, want, places=9):
        assert_complex_close(self, got, want, places=places)

    def test_fft_impulse(self):
        self.assert_complex_close(fft([1, 0, 0, 0]), [1, 1, 1, 1])

    def test_fft_constant(self):
        self.assert_complex_close(fft([1, 1, 1, 1]), [4, 0, 0, 0])

    def test_fft_known_values(self):
        # DFT of [0, 1, 0, -1] is [0, -2j, 0, 2j].
        self.assert_complex_close(fft([0, 1, 0, -1]), [0, -2j, 0, 2j])

    def test_fft_round_trip(self):
        original = [1 + 2j, 3 - 1j, -2 + 0.5j, 0, 4, -1j, 2, 2]
        self.assert_complex_close(ifft(fft(original)), original)

    def test_ifft_round_trip(self):
        original = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assert_complex_close(fft(ifft(original)), original)

    def test_fft_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            fft([1, 2, 3])
        with self.assertRaises(ValueError):
            fft([])

    def test_ifft_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            ifft([1, 2, 3])

    def test_fft_does_not_mutate_input(self):
        data = [1, 2, 3, 4]
        snapshot = list(data)
        fft(data)
        self.assertEqual(data, snapshot)

    def test_fft_length_one(self):
        self.assert_complex_close(fft([5 + 1j]), [5 + 1j])
        self.assert_complex_close(ifft([5 + 1j]), [5 + 1j])


class TestConvolution(unittest.TestCase):
    def assert_complex_close(self, got, want, places=9):
        assert_complex_close(self, got, want, places=places)

    def test_convolve_simple(self):
        self.assert_complex_close(convolve([1, 2, 3], [0, 1, 0.5]), [0, 1, 2.5, 4, 1.5])

    def test_convolve_matches_direct(self):
        a = [1, -2, 3, 0.5]
        b = [4, 1j, -1]
        expected = [0j] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                expected[i + j] += x * y
        self.assert_complex_close(convolve(a, b), expected)

    def test_convolve_empty(self):
        self.assertEqual(convolve([], [1, 2]), [])
        self.assertEqual(convolve([1, 2], []), [])

    def test_convolve_single_element(self):
        self.assert_complex_close(convolve([2], [3]), [6])

    def test_multiply_polynomials(self):
        # (1 + x)(1 - x) = 1 - x^2
        self.assert_complex_close(multiply_polynomials([1, 1], [1, -1]), [1, 0, -1])

    def test_multiply_polynomials_zero(self):
        self.assertEqual(multiply_polynomials([], [1, 2, 3]), [])


if __name__ == "__main__":
    unittest.main()
