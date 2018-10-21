from copy import copy
import unittest

import numpy as np

from bezier import Bezier, CtrlPoint, BezierError


class TestBezier(unittest.TestCase):

    def setUp(self):
        self.b0 = Bezier([(1, 1), (2, 3), (4, 3), (3, 1)])
        self.b1 = Bezier([(-1,1), (3,3), (4,-1), (2,-1)])

    def test_bezier(self):
        test = np.array([self.b0.point(i/10.0) for i in range(11)])
        ref = np.array([(1.000, 1.00),
                        (1.326, 1.54),
                        (1.688, 1.96),
                        (2.062, 2.26),
                        (2.424, 2.44),
                        (2.750, 2.50),
                        (3.016, 2.44),
                        (3.198, 2.26),
                        (3.272, 1.96),
                        (3.214, 1.54),
                        (3.000, 1.00)])
        self.assertTrue(np.allclose(test, ref))

    def test_eq(self):
        self.assertEqual(self.b0, copy(self.b0))

    def test_elevate(self):
        test1 = self.b1.elevate()
        ref1 = Bezier([(-1.0, 1.0), (2.0, 2.5), (3.5, 1.0), (3.5, -1.0), (2.0, -1.0)])

        test2 = self.b1.elevate(degree=2)
        ref2 = Bezier([(-1.0, 1.0), (1.4, 2.2), (2.9, 1.6), (3.5, 0.2), (3.2, -1.0), (2.0, -1.0)])

        self.assertEqual(test1, ref1)
        self.assertEqual(test2, ref2)

    def test_split(self):
        ref_l = Bezier([(-1.0, 1.0), (1.0, 2.0), (2.25, 1.5), (2.75, 0.75)])
        ref_r = Bezier([(2.75, 0.75), (3.25, 0.0), (3.0, -1.0), (2.0, -1.0)])

        test_l, test_r = self.b1.split(t=0.5)

        self.assertEqual(test_l, ref_l)
        self.assertEqual(test_r, ref_r)

    def test_curvature(self):
        ref = np.array([
            0.134164,
            0.219124,
            0.340136,
            0.471343,
            0.566666,
            0.625815,
            0.713340,
            0.904176,
            1.155625,
            1.095216,
            0.670820])
        test = np.array([self.b1.curvature(t=t) for t in np.linspace(0.0, 1.0, num=11)])

        self.assertTrue(np.allclose(test, ref))

    def test_derivated(self):
        b = Bezier([(-4.0, 0.00), (4.0, 0.00), (-2.0, 0.00), (2.0, 0.00)])
        d = b.derivative

        print(d.point(t=0.08))


if __name__ == '__main__':
    unittest.main()
