import unittest

import numpy as np
from numpy import linspace

from bezier import Bezier, fit_bezier


class TestFitBezier(unittest.TestCase):

    def test_fit_bezier(self):
        n = 100
        T = linspace(0.0, 1.0, n)
        b_ref = Bezier([(-1,1), (3,3), (4,-1), (2,-1)])

        for rtol in np.logspace(-5,-1, num=5):
            noise_pnts = np.array([b_ref.point(t) for t in T]) + rtol*(np.random.rand(n, 2) - 0.5)
            b_test = fit_bezier(noise_pnts, b_ref.deg)

            with self.subTest(rtol=rtol):
                self.assertTrue(np.allclose(
                    [p.point for p in b_test.ctrl_points],
                    [p.point for p in b_ref.ctrl_points],
                    rtol=rtol))


if __name__ == '__main__':
    unittest.main()
