import numpy as np
from scipy.special import binom
from scipy.linalg import norm, solve

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

    # @property
    # def derivative(self):
        # """Calculate derivative Bezier curve

        # References
        # ----------
        # .. [1] Thomas W. Sederberg, "COMPUTER AIDED GEOMETRIC DESIGN"
        #    BYU, p. 30, 2014.

        # """

        # # cpts = np.stack([p() for p in self.ctrl_points])

        # # print(np.diff(cpts, axis=0))

        # for i in range(self.deg):
        #     p1 = self.ctrl_points[i+1]
        #     p0 = self.ctrl_points[i]
        #     p = self.deg * (p1.weight / p0.weight) * (p1 - p0)
        #     print(p)

    # @property
    # def hodograph(self):
    #     """Calculate hodograph Bezier curve

    #     References
    #     ----------
    #     .. [1] Thomas W. Sederberg, "COMPUTER AIDED GEOMETRIC DESIGN"
    #        BYU, p. 31, 2014.

    #     """

    #     cpts = [np.array(self[i+1]) - np.array(self[i]) for i in range(self.deg)]

    #     return Bezier(cpts)

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

    # def nearest_point(self, point, t_guess=0.5, tol=0.000001):
        # """Estimate nearest pont on bezier using fmin func

        # Parameters
        # ----------
        # point : list
        #     point coordinates to evaluate
        # t_guess : float, optional
        #     inital t guess
        # tol : float, optional
        #     stopping tolerance
        # """

        # def error_fn(t_trial, p):
        #     return norm(p - self.point(t_trial[0]))

        # npoint = np.array(point)
        # t = sp.optimize.minimize(error_fn, t_guess, args=(npoint,), tol=tol)
        # return t['x'][0]
    # def nearest_point(self, point, t_guess=0.5, tol=0.000001):
        # """Estimate nearest pont on bezier using fmin func

        # Parameters
        # ----------
        # point : list
        #     point coordinates to evaluate
        # t_guess : float, optional
        #     inital t guess
        # tol : float, optional
        #     stopping tolerance
        # """

        # def error_fn(t_trial, p):
        #     return scipy.linalg.norm(p - self.point(t_trial[0]))

        # npoint = np.array(point)
        # t = scipy.optimize.minimize(error_fn, t_guess, args=(npoint,), tol=tol)
        # return t['x'][0]

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


def fit_bezier(pnts, deg):
    if int(deg) < 1:
        raise ValueError('Bezier degree must be greater than 1')
    else:
        deg = int(deg)

    pnts = np.array(pnts)

    def solve_for_cs(ds, ts, deg):
        """Takes an input series of values (ds) and uses them to solve Ax = b
        ds = values
        ts = time steps between ds

        """

        # Create A
        #   am, bm = [1, 4, 6, 4, 1] (binomial array for deg 4)
        am = np.array([[binom(deg, i) for i in range(deg + 1)]])
        bm = np.array([[binom(deg, i) for i in range(deg + 1)]]).T

        # cm = matrix of 1's and -1's (based on index, e.g. for even deg: 0,0 = 1; 0,1 = -1;, 0,2 = 1, etc)
        cm = (1 - 2*np.mod(np.sum(np.indices((deg+1, deg+1)), axis=0), 2*np.ones((deg+1, deg+1))))

        # Fix for odd deg (the array needs to be flipped... -1, 1 instead of 1, -1)
        if deg % 2:
            cm *= -1

        # A is square binomial matrix scaled by matrix of 1's and -1s (scalar of two from differentiation)
        A = 2*am*bm*cm

        # Create B
        #   B is matrix of exponents for (t-1) [[8, 7, 6, 5, 4], [7, 6, 5, 4, 3], ...[4, 3, 2, 1, 0]]
        B = np.zeros((deg+1, deg+1))
        B[-1] = np.arange(deg, -1, -1)

        for i in range(deg - 1, -1, -1):
            B[i] = B[i+1] + 1

        # Create C
        #   C is matrix of exponents for t [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], ...[4, 5, 6, 7, 8]]
        C = np.zeros((deg+1, deg+1))
        C[0] = np.arange(deg + 1)

        for i in range(deg):
            C[i+1] = C[i] + 1

        # Create D
        #   am is 1 row matrix of binomial coefficients
        am = np.array([binom(deg, i) for i in range(deg + 1)])

        # bm is matrix of 1's and -1's (based on index)
        bm = (-1 + 2*np.mod(np.sum(np.indices((1, deg+1)), axis=0).flatten(), 2 * np.ones((deg+1))))

        # D is binomial coefficients scaled by matrix of 1's and -1's (scalar of two from differentiation)
        D = 2*am*bm

        # Create E
        E = np.arange(deg, -1, -1)

        # Create F
        F = np.arange(deg + 1)

        # Function to create A matrix in Ax=b (using each input from above (A, B, C))
        def A_fn(ts, A_, B_, C_):
            return sum([A_ * (t - 1) ** B_ * t ** C_ for t in ts])

        def b_fn(ts, ds, D_, E_, F_):
            return -1.0 * sum([D_ * (t - 1) ** E_ * d * t ** F_ for t, d in zip(ts, ds)])

        # Matrix A
        A = -1*np.array([A_fn(ts, A_, B_, C_) for A_, B_, C_ in
            zip(A.flatten(), B.flatten(), C.flatten())]).reshape((deg+1, deg+1))

        # Vector b
        b = -1*np.array([b_fn(ts, ds, D_, E_, F_) for D_, E_, F_ in
            zip(D.flatten(), E.flatten(), F.flatten())]).reshape((deg+1, 1))

        # Solve Ax = b
        return solve(A, b).ravel()

    ts = np.linspace(0.0, 1, len(pnts))

    # Fit spline through each direction independently
    ctrl_points = np.apply_along_axis(solve_for_cs, 0, pnts, ts, deg)

    return Bezier(ctrl_points)