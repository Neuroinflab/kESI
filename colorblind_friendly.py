# Based on
# Color Universal Design (CUD)
# - How to make figures and presentations that are friendly to Colorblind people
#
#
# Masataka Okabe
# Jikei Medial School (Japan)
#
# Kei Ito
# University of Tokyo, Institute for Molecular and Cellular Biosciences (Japan)
# (both are strong protanopes)
# 11.20.2002 (modified on 2.15.2008, 9.24.2008)
# http://jfly.iam.u-tokyo.ac.jp/color/#pallet

import numpy as np
from matplotlib import colors


class _sRGB(object):
    __slots__ = ('_r', '_g', '_b')

    _CIE_1931_XYZ_to_lRGB = [[3.240_625_5, -1.537_208_0, -0.498_628_6],
                             [-0.968_930_7, 1.875_756_1, 0.041_517_5],
                             [0.055_710_1, -0.204_021_1, 1.056_995_9]]

    def __init__(self, r, g, b):
        self._r = r
        self._g = g
        self._b = b

    def __str__(self):
        return '#{}'.format(''.join('{:02X}'.format(self._as_unsigned_char(v))
                                    for v in [self._r, self._g, self._b]))

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       repr(self._r),
                                       repr(self._g),
                                       repr(self._b))

    def __add__(self, other):
        return self.from_lRGB(*np.minimum(1, self.lRGB + other.lRGB))

    def __mul__(self, other):
        return self.from_lRGB(*np.minimum(1, self.lRGB * other))

    def __truediv__(self, other):
        return self.from_lRGB(*np.minimum(1, self.lRGB / other))

    @classmethod
    def from_unsigned_char(cls, r, g, b):
        return cls(r / 255.,
                   g / 255.,
                   b / 255.)

    @staticmethod
    def _as_unsigned_char(v):
        return int(round(v * 255))

    @staticmethod
    def _to_linear(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    @staticmethod
    def _from_linear(v):
        return v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1/2.4)) - 0.055

    @property
    def lRGB(self):
        return np.array([self._to_linear(v)
                         for v in self])

    @property
    def CIE_1931_XYZ(self):
        return np.linalg.solve(self._CIE_1931_XYZ_to_lRGB,
                               self.lRGB)

    @classmethod
    def from_lRGB(cls, r, g, b):
        return cls(*map(cls._from_linear,
                        (r, g, b)))

    @property
    def red(self):
        return self._r

    @property
    def green(self):
        return self._g

    @property
    def blue(self):
        return self._b

    def __len__(self):
        return 3

    def __iter__(self):
        return iter((self._r,
                     self._g,
                     self._b))


_BLACK     = _sRGB.from_unsigned_char(0, 0, 0)
_ORANGE    = _sRGB.from_unsigned_char(230, 159, 0)
_SKY_BLUE  = _sRGB.from_unsigned_char(86, 180, 233)
_GREEN     = _sRGB.from_unsigned_char(0, 158, 115)
_YELLOW    = _sRGB.from_unsigned_char(240, 228, 66)
_BLUE      = _sRGB.from_unsigned_char(0, 114, 178)
_VERMILION = _sRGB.from_unsigned_char(213, 94, 0)
_PURPLE    = _sRGB.from_unsigned_char(204, 121, 167)

BLACK     = str(_BLACK)
ORANGE    = str(_ORANGE)
SKY_BLUE  = str(_SKY_BLUE)
GREEN     = str(_GREEN)
YELLOW    = str(_YELLOW)
BLUE      = str(_BLUE)
VERMILION = str(_VERMILION)
PURPLE    = str(_PURPLE)


def _BipolarColormap(name, negative, positive):
    neg_max = negative.lRGB.max()
    pos_max = positive.lRGB.max()
    neg_Y = negative.CIE_1931_XYZ[1]
    pos_Y = positive.CIE_1931_XYZ[1]

    neg_scale = 0.5 * (neg_Y + pos_Y) / neg_Y
    pos_scale = 0.5 * (neg_Y + pos_Y) / pos_Y

    if neg_scale * neg_max > 1:
        pos_scale /= neg_max * neg_scale
        # neg_scale /= neg_max * neg_scale
        neg_scale = 1. / neg_max

    if pos_scale * pos_max > 1:
        neg_scale /= pos_max * pos_scale
        # pos_scale /= pos_max * pos_scale
        pos_scale = 1. / pos_max

    negative = negative * neg_scale
    positive = positive * pos_scale
    return colors.LinearSegmentedColormap(
                      name,
                      {k: [(0.0,) + (getattr(negative, k),) * 2,
                           (0.5, 1.0, 1.0),
                           (1.0,) + (getattr(positive, k),) * 2,
                           ]
                       for k in ['red', 'green', 'blue']})


bwr = _BipolarColormap('cbf.bwr', _BLUE, _ORANGE)
PRGn = _BipolarColormap('cbf.PRGn', _PURPLE, _GREEN)
