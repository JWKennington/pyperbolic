"""Define a point in hyperbolic space 

"""


import functools
import numpy
import typing
from pyperbolic import util


def as_points(f):
    @functools.wraps(f)
    def wrapper(*ps):
        return f(*(coerce_to_point(p) for p in ps))
    return wrapper


class Point:
    __slots__ = ('_coords', '_dim', '_arr')

    def __init__(self, *coords: typing.Tuple[float]):
        if isinstance(coords[0], numpy.ndarray):
            arr = coords[0]
            if not arr.ndim == 1:
                raise util.PyperbolicError('Can only construct Point from numpy.ndarray of ndim = 1, not {}'.format(arr.ndim))
            self._coords = arr
        self._dim = len(coords)
        self._arr = numpy.array(coords)

    @as_points
    def __add__(self, other):
        return Point.from_array(*self._coords + other._coords)

    @property
    def coords(self):
        return self._coords

    @property
    def dim(self):
        return self._dim

    @staticmethod
    def from_array(arr: numpy.ndarray):
        
        return Point(*arr.tolist())


def coerce_to_point(p: typing.Union[tuple, numpy.ndarray, Point]) -> Point:
    if isinstance(p, Point):
        pass
    elif isinstance(p, tuple):
        p = Point(*p)
    elif isinstance(p, numpy.ndarray):
        p = Point.from_array(p)
    else:
        raise util.PyperbolicError('Unable to coerce {} of type {} to Point'.format(p, type(p)))
    return p
