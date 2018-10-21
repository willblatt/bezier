import numpy as np
from scipy.special import binom
from scipy.linalg import solve

from . import Bezier


def fit_bezier(pnts, deg):
    if len(pnts) < 2:
        raise ValueError('At least 2 points are needed to fit')
    pnts = np.array(pnts)

    if int(deg) < 1:
        raise ValueError('Bezier degree must be greater than 1')
    deg = int(deg)

    T = np.linspace(0.0, 1, len(pnts))

    def solve_for_cs(D, T, deg):
        """Takes an input series of values (D) and uses them to solve Ax = b
        D = values
        T = spacing between D

        """

        # Create Q
        #   am = [1, 4, 6, 4, 1] (binomial array for deg 4)
        am = np.array([[binom(deg, i) for i in range(deg + 1)]])

        # cm = matrix of 1's and -1's (based on index, e.g. for even deg: 0,0 = 1; 0,1 = -1;, 0,2 = 1, etc)
        cm = (1 - 2*np.mod(np.sum(np.indices((deg+1, deg+1)), axis=0), 2*np.ones((deg+1, deg+1))))

        # Fix for odd deg (the array needs to be flipped... -1, 1 instead of 1, -1)
        if deg % 2:
            cm *= -1

        # A is square binomial matrix scaled by matrix of 1's and -1s (scalar of two from differentiation)
        Q = 2*am*am.T*cm

        # Create R
        #   R is matrix of exponents for (t-1) [[8, 7, 6, 5, 4], [7, 6, 5, 4, 3], ...[4, 3, 2, 1, 0]]
        R = np.flip(np.arange(deg+1) + np.arange(deg+1)[:,None])

        # Create S
        #   S is matrix of exponents for t [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], ...[4, 5, 6, 7, 8]]
        S = np.arange(deg+1) + np.arange(deg+1)[:,None]

        # Matrix A
        A = -1 * np.array(
            [np.sum(q * (T-1)**r * T**s) for q, r, s in zip(Q.ravel(), R.ravel(), S.ravel())]
        ).reshape((deg+1, deg+1))

        #   am is 1 row matrix of binomial coefficients
        am = np.array([binom(deg, i) for i in range(deg + 1)])

        # bm is matrix of 1's and -1's (based on index)
        bm = (-1 + 2*np.mod(np.sum(np.indices((1, deg+1)), axis=0).ravel(), 2 * np.ones((deg+1))))

        # U is binomial coefficients scaled by matrix of 1's and -1's (scalar of two from differentiation)
        U = 2*am*bm

        V = np.arange(deg, -1, -1)
        W = np.arange(deg + 1)

        # Vector b
        b = np.array(
            [np.sum(u * (T-1)**v * D*T**w) for u, v, w in zip(U.ravel(), V.ravel(), W.ravel())]
        ).reshape((deg+1, 1))

        # Solve Ax = b
        return solve(A, b).ravel()

    # Fit spline through each direction independently
    ctrl_points = np.apply_along_axis(solve_for_cs, 0, pnts, T, deg)

    return Bezier(ctrl_points)
