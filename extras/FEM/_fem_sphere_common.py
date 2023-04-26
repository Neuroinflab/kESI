#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2020 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################

import logging

import numpy as np


try:
    from . import _fem_common as fc
    # When run as script raises:
    #  - `ImportError` (Python 3.6-9), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except ImportError:
    import _fem_common as fc


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _SourceBase(object):
    def __init__(self, r, altitude, azimuth, parent):
        self._r = r
        self._altitude = altitude
        self._azimuth = azimuth
        self.parent = parent

        sin_alt = np.sin(self._altitude)  # np.cos(np.pi / 2 - altitude)
        cos_alt = np.cos(self._altitude)  # np.sin(np.pi / 2 - altitude)
        sin_az = np.sin(self._azimuth)  # -np.sin(-azimuth)
        cos_az = np.cos(self._azimuth)  # np.cos(-azimuth)

        self._apply_trigonometric_functions(cos_alt, sin_alt, cos_az, sin_az)

    def _apply_trigonometric_functions(self, cos_alt, sin_alt, cos_az, sin_az):
        r2 = self._r * cos_alt
        self._x = r2 * cos_az
        self._y = r2 * sin_az
        self._z = self._r * sin_alt

    def _RPI_as_LIP(self, X, Y, Z):
        return -X, Z, Y

    def _LIP_as_RPI(self, X, Y, Z):
        return -X, Z, Y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def r(self):
        return self._r

    @property
    def altitude(self):
        return self._altitude

    @property
    def azimuth(self):
        return self._azimuth


class _RotatingSourceBase(_SourceBase):
    def _apply_trigonometric_functions(self, cos_alt, sin_alt, cos_az, sin_az):
        super(_RotatingSourceBase,
              self)._apply_trigonometric_functions(cos_alt, sin_alt, cos_az, sin_az)

        # As the source is to be rotated by az, the electrodes
        # needs to be rotated by -az.
        # sin(-az) = -sin(az)
        # cos(-az) = cos(az)
        ROT_XY = [[cos_az, -sin_az, 0],
                  [sin_az, cos_az, 0],
                  [0, 0, 1]]

        # As the original source is at (0, 0, R), its altitude
        # is already pi/2.  The source needs to be rotated by
        # pi/2 - alt, thus the electrodes needs to be rotated
        # alt - pi / 2.
        # sin(alt - pi/2) == -cos(alt)
        # cos(alt - pi/2) == sin(alt)
        ROT_ZX = [[sin_alt, 0, cos_alt],
                  [0, 1, 0],
                  [-cos_alt, 0, sin_alt]]
        # As the source is raotated by altitude, then by azimuth,
        # the electrodes are to be rotated in inverse order
        self._ROT = np.matmul(ROT_XY,
                              ROT_ZX)

    def _rotated(self, X, Y, Z):
        _X = self._ROT[0, 0] * X + self._ROT[1, 0] * Y + self._ROT[2, 0] * Z
        _Y = self._ROT[0, 1] * X + self._ROT[1, 1] * Y + self._ROT[2, 1] * Z
        _Z = self._ROT[0, 2] * X + self._ROT[1, 2] * Y + self._ROT[2, 2] * Z
        return self._RPI_as_LIP(_X, _Y, _Z)

    def potential(self, X, Y, Z):
        _X, _Y, _Z = self._rotated(X, Y, Z)
        return self._potential_rotated(_X, _Y, _Z)
