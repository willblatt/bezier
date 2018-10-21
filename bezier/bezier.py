import numpy as np
from scipy import optimize
from scipy.linalg import norm
from scipy.special import binom

from .ctrl_point import CtrlPoint


class BezierError(Exception):
    pass


class Bezier:

    def __init__(self, points):
        self.ctrl_points = points

    def __repr__(self):
        return f"{self.ctrl_points}"

    def __setitem__(self, idx, point):
        self.ctrl_points[idx] = CtrlPoint(point)

    def __getitem__(self, idx):
        return self.ctrl_points[idx]

    def __eq__(self, other):

        if not isinstance(other, Bezier):
            return False

        if self.dim != other.dim:
            return False

        if self.deg != other.deg:
            return False

        for p, q in zip(self.ctrl_points, other.ctrl_points):
            if p != q:
                return False

        return True

    @property
    def dim(self):
        return self.ctrl_points[0].dim

    @property
    def deg(self):
        return len(self.ctrl_points) - 1

    @property
    def ctrl_points(self):
        return self._ctrl_points

    @ctrl_points.setter
    def ctrl_points(self, points):
        self._ctrl_points = [CtrlPoint(p) for p in points]

    @property
    def derivative(self):
        """Calculate derivative Bezier curve

        References
        ----------
        .. [1] Thomas W. Sederberg, "COMPUTER AIDED GEOMETRIC DESIGN"
           BYU, p. 30, 2014.

        """

        # cpts = np.stack([p() for p in self.ctrl_points])

        # print(np.diff(cpts, axis=0))

        cps = []
        for i in range(self.deg):
            p1 = self.ctrl_points[i+1]
            p0 = self.ctrl_points[i]
            p = (self.deg-1) * (p1.weight / p0.weight) * (p1 - p0)
            cps.append(p)

        return Bezier(cps)

    def point(self, t):
        """Evaluate curve at a given t value"""

        p = np.zeros(self.dim)
        d = 0.0
        for i, ctrl_point in enumerate(self.ctrl_points):
            b = ctrl_point.weight * binom(self.deg, i) * (1-t)**(self.deg-i) * t**(i)
            p += b * ctrl_point.point
            d += b
        p /= d

        return p

    def elevate(self, degree=1):
        """Create a degree elevated Bezier curve

        Parameters
        ----------
        degree : int > 0
            Number of degrees to elevate

        Notes
        -----
        .. math:: P$^*_i$ = \alpha P_{i-1} + (1 - \alpha) P_i

        where α is defined as:

        .. math:: \alpha_i = \frac{i}{n+1}

        References
        ----------
        .. [1] Thomas W. Sederberg, "COMPUTER AIDED GEOMETRIC DESIGN"
           BYU, p. 23, 2014.

        """

        if degree <= 0:
            raise ValueError('degree elevation increment must be positive')

        points = [p.point for p in self.ctrl_points]

        for i in range(degree):
            p0 = np.insert(np.array(points), 0, [points[0]], axis=0)
            p1 = np.append(np.array(points), [points[0]], axis=0)

            a = np.arange(len(points)+1) / (self.deg+1 +i)

            points = p0*a[:,None] + p1*(1-a)[:,None]

        return Bezier(points)

    def nearest(self, point, t_guess=0.5, tol=0.000001):
        """Estimate nearest pont on bezier using fmin func

        Parameters
        ----------
        point : list
            point coordinates to evaluate
        t_guess : float, optional
            inital t guess
        tol : float, optional
            stopping tolerance
        """

        def error_fn(t_trial, p):
            return norm(p - self.point(t_trial[0]))

        point = np.array(point)
        t = optimize.minimize(error_fn, t_guess, args=(point,), tol=tol)

        return t['x'][0]

    def split(self, t):
        """Performs the de Casteljau algorithm to split the current Bezier
        at a given t value.

        Notes
        -----
        Use the De Casteljau's method to split the Bezier at a given t value
        and then, using the end points, calculate:

        .. math:: \kappa = \frac{n-1}{n}\frac{h}{a^2}

        where a is the length of the first leg of the control polygon,
        and h is the projected distance of the 3rd point to the first leg

        References
        ----------
        .. [1] Thomas W. Sederberg, "COMPUTER AIDED GEOMETRIC DESIGN"
           BYU, p. 31, 2014.

        """

        # Create containers for output beziers
        l_points, r_points = [], []

        # Use the original control points as starting points
        for i in range(self.deg+1):
            l_points.append(np.array(self.ctrl_points[i].point))
            r_points.append(np.array(self.ctrl_points[i].point))

        l_points.reverse()

        # Walk through and perform de Casteljau's algorithm
        for i in range(self.deg):
            for j in range(self.deg-i):
                r_points[j] = t*r_points[j+1] + (1-t)*r_points[j]
                l_points[j] = t*l_points[j] + (1-t)*l_points[j+1]

        l_points.reverse()

        return Bezier(l_points), Bezier(r_points)

    def curvature(self, t):
        """Calculates Curvature at value t

        Notes
        -----
        Use the De Casteljau's method to split the Bezier at a given t value
        and then, using the end points, calculate:

        .. math:: \kappa = \frac{n-1}{n}\frac{h}{a^2}

        where a is the length of the first leg of the control polygon,
        and h is the projected distance of the 3rd point to the first leg

        References
        ----------
        .. [1] Thomas W. Sederberg, "COMPUTER AIDED GEOMETRIC DESIGN"
           BYU, p. 31, 2014.

        """

        def _h(t, b):
            p0 = np.array(b.ctrl_points[0])
            p1 = np.array(b.ctrl_points[1])
            p2 = np.array(b.ctrl_points[2])
            q = (p0 - p1).point
            r = (p1 - p2).point
            return norm(np.cross(q, r)) / norm(q)

        left, right = self.split(t)

        # Use right bezier, unless t=1.0
        if t < 1.0:
            h = _h(t, right)
            a = norm(right.ctrl_points[0].point - right.ctrl_points[1].point)
        else:
            h = _h(t, left)
            a = norm(left.ctrl_points[-1].point - left.ctrl_points[-2].point)

        n = float(self.deg)

        return ((n-1.0)/n) * h/(a**2.0)
