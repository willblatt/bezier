import unittest

import numpy as np

from bezier import CtrlPoint


class TestCtrlPoint(unittest.TestCase):

    def setUp(self):
        self.p0 = np.array([1, 2, 3])
        self.p1 = np.array([3, 1, 3])

    def test_create_ctrl_point(self):
        weight = 1.1
        cp = CtrlPoint(self.p0, weight=weight)

        self.assertTrue(np.array_equal(cp.point, self.p0))
        self.assertEqual(cp.weight, weight)
        self.assertEqual(cp.dim, len(self.p0))
        self.assertEqual(CtrlPoint(self.p0.reshape(3,1)), CtrlPoint(self.p0))

    def test_eq(self):
        self.assertTrue(CtrlPoint(self.p0) == CtrlPoint(self.p0))

    def test_add(self):
        self.assertEqual(CtrlPoint(self.p1) + CtrlPoint(self.p0), self.p1 + self.p0)

    def test_sub(self):
        self.assertEqual(CtrlPoint(self.p1) - CtrlPoint(self.p0), self.p1 - self.p0)

    def test_mul(self):
        self.assertEqual(3*CtrlPoint(self.p0), CtrlPoint(3*self.p0))
        self.assertEqual(CtrlPoint(self.p0)*3, CtrlPoint(self.p0*3))

    def test_div(self):
        self.assertEqual(CtrlPoint(self.p0)/3, CtrlPoint(self.p0/3))

    def test_dim(self):
        pass


if __name__ == '__main__':
    unittest.main()
