import numpy as np


class CtrlPoint:
    """Bezier control point"""

    def __init__(self, point, weight=1.0):
        self.point = point
        self.weight = weight

    def __repr__(self):
        return f"CtrlPoint({self.point}, weight={self.weight})"

    def __eq__(self, other):
        p = other.point if isinstance(other, self.__class__) else np.array(other)
        w = other.weight if isinstance(other, self.__class__) else 1.0

        if np.allclose(self.point, p) and self.weight == w:
            return True
        else:
            return False

    def __add__(self, other):
        return self.__class__([a + b for a, b in zip(self.point, self.__class__(other).point)])

    def __sub__(self, other):
        return self.__class__([a - b for a, b in zip(self.point, self.__class__(other).point)])

    def __mul__(self, other):
        return self.__class__([a*other for a in self.point], weight=self.weight)

    def __rmul__(self, other):
        return self.__class__([a*other for a in self.point], weight=self.weight)

    def __truediv__(self, other):
        return self.__class__([a/other for a in self.point], weight=self.weight)

    @property
    def dim(self):
        return self._point.size

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value):
        if isinstance(value, self.__class__):
            self._point = value.point
        else:
            self._point = np.array(value, dtype='=f8').flatten()

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = float(value)
