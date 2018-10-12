import unittest

import numpy as np

from bezier import CtrlPoint, Bezier


class TestCtrlPoint(unittest.TestCase):

    def test_ctrl_point(self):
        p = [1, 2, 3]
        weight = 1.1
        cp = CtrlPoint(p, weight=weight)

        self.assertTrue(np.array_equal(cp.point, np.array(p)))
        self.assertEqual(cp.weight, weight)
        self.assertEqual(cp.dim, len(p))

    def test_add(self):
        p0 = np.array([1, 2, 3])
        p1 = np.array([3, 1, 3])

        self.assertEqual(CtrlPoint(p0) + CtrlPoint(p1), CtrlPoint(p1 + p0))

    def test_sub(self):
        p0 = np.array([1, 2, 3])
        p1 = np.array([3, 1, 3])

        self.assertEqual(CtrlPoint(p1) - CtrlPoint(p0), CtrlPoint(p1 - p0))


class TestBezier(unittest.TestCase):

    def test_bezier(self):
        b = Bezier([(1, 1), (2, 3), (4, 3), (3, 1)])

        test = np.array([b.point(i/10.0) for i in range(11)])
        ref = np.array([[1.000, 1.00],
                        [1.326, 1.54],
                        [1.688, 1.96],
                        [2.062, 2.26],
                        [2.424, 2.44],
                        [2.750, 2.50],
                        [3.016, 2.44],
                        [3.198, 2.26],
                        [3.272, 1.96],
                        [3.214, 1.54],
                        [3.000, 1.00]])

        np.array_equal(test, ref)

        b.elevate(1)

#     def test_curv(self):
#         p0 = CtrlPoint((0, 0), weight=1)
#         p1 = CtrlPoint((4, 3), weight=2)
#         p2 = CtrlPoint((0, 5), weight=4)

#         b = Bezier((p0, p1, p2))

#         b.curvature(t=0)