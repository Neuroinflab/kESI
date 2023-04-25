#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

logger = logging.getLogger(__name__)


class GaussKronrodSourceIntegrator(object):
    """
    An experimental source based on numerical integration
    """
    # Gauss–Kronrod quadrature nodes and weights copied from:
    #   https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
    # which references paragraph 5.5 of:
    #   Kahaner, David; Moler, Cleve; Nash, Stephen (1989)
    #   Numerical Methods and Software, Prentice–Hall, ISBN 978-0-13-627258-8
    _NODES = [
              0.99145_53711_20813,
              0.94910_79123_42759,
              0.86486_44233_59769,
              0.74153_11855_99394,
              0.58608_72354_67691,
              0.40584_51513_77397,
              0.20778_49550_07898,
              0.00000_00000_00000,
              ]

    _GAUSS = [
              0.00000_00000_00000,
              0.12948_49661_68870,
              0.00000_00000_00000,
              0.27970_53914_89277,
              0.00000_00000_00000,
              0.38183_00505_05119,
              0.00000_00000_00000,
              0.41795_91836_73469,
              ]

    _KRONROD = [
                0.02293_53220_10529,
                0.06309_20926_29979,
                0.10479_00103_22250,
                0.14065_32597_15525,
                0.16900_47266_39267,
                0.19035_05780_64785,
                0.20443_29400_75298,
                0.20948_21410_84728,
                ]

    NODES = np.array([-x for x in _NODES[:-1]] + _NODES[::-1])
    _W_GAUSS = np.array(_GAUSS[:-1] + _GAUSS[::-1])
    _W_KRONROD = np.array(_KRONROD[:-1] + _KRONROD[::-1])

    _WX, _WY, _WZ = np.meshgrid(_W_GAUSS, _W_GAUSS, _W_GAUSS)
    GAUSS = (_WX * _WY * _WZ).flatten()
    _WX, _WY, _WZ = np.meshgrid(_W_KRONROD, _W_KRONROD, _W_KRONROD)
    KRONROD = (_WX * _WY * _WZ).flatten()

    ERROR = GAUSS - KRONROD

    def __init__(self,
                 csd,
                 leadfield,
                 limits):
        self._csd = csd
        self._leadfield = leadfield

        _max = np.max(limits, axis=1)
        _min = np.min(limits, axis=1)

        (self._x_max,
         self._y_max,
         self._z_max) = _max

        (self._x_min,
         self._y_min,
         self._z_min) = _min

        _mean = np.mean(limits, axis=1)
        (self._x_mean,
         self._y_mean,
         self._z_mean) = _mean
        (self._x_r,
         self._y_r,
         self._z_r) = _max - _mean

        X_GRID = self.NODES * self._x_r + self._x_mean
        Y_GRID = self.NODES * self._y_r + self._y_mean
        Z_GRID = self.NODES * self._z_r + self._z_mean
        X, Y, Z = np.meshgrid(X_GRID,
                              Y_GRID,
                              Z_GRID)

        self.X, self.Y, self.Z = X.flatten(), Y.flatten(), Z.flatten()

        CSD = self._csd(self.X, self.Y, self.Z).flatten() * (self._x_r * self._y_r * self._z_r)
        # self._CSD_GAUSS = CSD * self.GAUSS
        self._CSD_KRONROD = CSD * self.KRONROD
        self._CSD_ERROR = CSD * self.ERROR

    def csd(self, X, Y, Z):
        return np.where(((X >= self._x_min)
                         & (X <= self._x_max)
                         & (Y >= self._y_min)
                         & (Y <= self._y_max)
                         & (Z >= self._z_min)
                         & (Z <= self._z_max)),
                        self.csd(X, Y, Z),
                        0)

    def potential(self, *args, **kwargs):
        LF = self._leadfield(self.X,
                             self.Y,
                             self.Z,
                             *args,
                             **kwargs)
        return self._integrate_potential(LF, self._CSD_KRONROD)

    def _integrate_potential(self, LEADFIELD, WEIGHTED_CSD):
        shape = (-1,) + (1,) * (len(LEADFIELD.shape) - 1)
        return (LEADFIELD * WEIGHTED_CSD.reshape(shape)).sum(axis=0)

    def error(self, *args, **kwargs):
        LF = self._leadfield(self.X,
                             self.Y,
                             self.Z,
                             *args,
                             **kwargs)
        return np.abs(self._integrate_potential(LF, self._CSD_ERROR))


class IntegratedGaussianSourceKCSD3D(object):
    def __init__(self, x, y, z, standard_deviation, conductivity):
        self.x = x
        self.y = y
        self.z = z
        self._variance = standard_deviation ** 2
        self._a = (2 * np.pi * self._variance) ** -1.5
        self.conductivity = conductivity
        r = standard_deviation * 3

        self._quadrature = GaussKronrodSourceIntegrator(self._csd,
                                                        self._leadfield,
                                                        [[x - r, x + r],
                                                                   [y - r, y + r],
                                                                   [z - r, z + r]])

    def _leadfield(self, X, Y, Z, EX, EY, EZ):
        shape_e = (1,) + EX.shape
        shape = (-1,) + (1,) * len(EX.shape)
        return ((0.25 / (np.pi * self.conductivity))
                / np.sqrt(self._distance2(X.reshape(shape),
                                          Y.reshape(shape),
                                          Z.reshape(shape),
                                          EX.reshape(shape_e),
                                          EY.reshape(shape_e),
                                          EZ.reshape(shape_e))))
        # may misbehave if quadrature node hits electrode

    def _csd(self, X, Y, Z):
        x0 = self.x
        y0 = self.y
        z0 = self.z
        return self._a * np.exp(-0.5 * self._distance2(X, Y, Z, x0, y0,
                                                       z0) / self._variance)

    def _distance2(self, X, Y, Z, x0, y0, z0):
        return (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2

    def csd(self, X, Y, Z):
        return self._quadrature.csd(X, Y, Z)

    def potential(self, X, Y, Z):
        return self._quadrature.potential(X, Y, Z)

    def error(self, X, Y, Z):
        return self._quadrature.error(X, Y, Z)


if __name__ == '__main__':
    from kesi.common import GaussianSourceKCSD3D
    import cbf
    import matplotlib.pyplot as plt

    CONDUCTIVITY = 0.3  # S / m
    H = 0.15e-3         # m
    SD = H / 8          # m
    N = 100


    X, Y, Z = np.meshgrid(np.linspace(-H, H, N + 1),
                          np.linspace(-H, H, N + 1),
                          [0])

    exact = GaussianSourceKCSD3D(0, 0, 0, SD, CONDUCTIVITY)
    numeric = IntegratedGaussianSourceKCSD3D(0, 0, 0, SD, CONDUCTIVITY)

    EXACT = np.ma.masked_invalid(exact.potential(X, Y, Z))
    NUMERIC = np.ma.masked_invalid(numeric.potential(X, Y, Z))
    ERROR = np.ma.masked_invalid(numeric.error(X, Y, Z))


    FACTOR_X = 1e3
    FACTOR_POTENTIAL = 4e-5 # physiological CSD
    potential_max = max(abs(EXACT).max(),
                        abs(NUMERIC).max(),
                        abs(EXACT - NUMERIC).max(),
                        ERROR.max()) * FACTOR_POTENTIAL
    potential_levels = np.linspace(-potential_max, potential_max, 256)
    potential_ticks_major = np.linspace(-potential_max, potential_max, 6)
    XTICKS = YTICKS = np.linspace(-H, H, 7) * FACTOR_X
    XTICKS_MINOR = YTICKS_MINOR = []

    def plot_potential(POTENTIAL, title=None):
        fig, (ax, cax) = plt.subplots(1, 2,
                                      gridspec_kw={'width_ratios': [9, 1],
                                                   })
        if title is not None:
            ax.set_title(title)

        ax.set_aspect("equal")
        im = ax.contourf(X[:,:,0] * FACTOR_X,
                         Y[:,:,0] * FACTOR_X,
                         POTENTIAL[:,:,0] * FACTOR_POTENTIAL,
                         levels=potential_levels,
                         cmap=cbf.PRGn)
        ax.contour(X[:,:,0] * FACTOR_X,
                   Y[:,:,0] * FACTOR_X,
                   POTENTIAL[:,:,0] * FACTOR_POTENTIAL,
                   levels=potential_ticks_major,
                   colors=cbf.BLACK,
                   linestyles=':')
        fig.colorbar(im,
                     cax=cax,
                     orientation='vertical',
                     format='%.3g',
                     ticks=potential_ticks_major)

        cax.minorticks_on()
        cax.set_ylabel('$V$')

        ax.set_ylim(-H * FACTOR_X, H * FACTOR_X)
        ax.set_xlim(-H * FACTOR_X, H * FACTOR_X)
        ax.set_xticks(XTICKS)
        ax.set_xticks(XTICKS_MINOR,
                      minor=True)
        ax.set_yticks(YTICKS)
        ax.set_yticks(YTICKS_MINOR,
                      minor=True)
        ax.set_xlabel('$mm$')
        ax.set_ylabel('$mm$')
        return fig, ax

    plot_potential(EXACT, 'Exact')
    plot_potential(NUMERIC, 'Numeric')
    fig, ax = plot_potential(NUMERIC, 'Numeric & nodes')
    ax.scatter(numeric._quadrature.X[numeric._quadrature.Z == 0] * FACTOR_X,
               numeric._quadrature.Y[numeric._quadrature.Z == 0] * FACTOR_X,
               marker='x',
               color=cbf.BLACK)
    plot_potential(ERROR, 'Estimated error')
    plot_potential(NUMERIC - EXACT, 'Numeric - Exact')
    fig, ax = plot_potential(NUMERIC - EXACT, 'Numeric - Exact & nodes')
    ax.scatter(numeric._quadrature.X[numeric._quadrature.Z == 0] * FACTOR_X,
               numeric._quadrature.Y[numeric._quadrature.Z == 0] * FACTOR_X,
               marker='o',
               edgecolors=cbf.BLACK,
               facecolor='none')
    print("""The error is high near nodes of the quadrature.
It is due to the high leadfield value (due to short distance).

To avoid this artifact potential nodes need to be moved away from the quadrature
nodes.  It is possible that Newton-Cotes quadratures are more useful as they
use uniformly distributed nodes, which may be easily shared between sources,
thus simplifying separation of potential nodes from quadrature nodes.
""")
    plt.show()