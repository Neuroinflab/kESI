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
import collections
import operator
import warnings

import numpy as np
from scipy.special import erf, lpmv


logger = logging.getLogger(__name__)


class Gauss3D(object):
    __slots__ = ('x', 'y', 'z', 'standard_deviation', '_variance', '_a')

    def __init__(self, x, y, z, standard_deviation):
        self.x = x
        self.y = y
        self.z = z
        self.standard_deviation = standard_deviation
        self._variance = standard_deviation ** 2
        self._a = (2 * np.pi * self._variance) ** -1.5

    def __call__(self, X, Y, Z):
        return self._a * np.exp(-0.5 * (np.square(X - self.x)
                                        + np.square(Y - self.y)
                                        + np.square(Z - self.z)) / self._variance)


class SourceBase(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class GaussianSourceBase(SourceBase):
    def __init__(self, x, y, z, standard_deviation):
        super(GaussianSourceBase, self).__init__(x, y, z)
        self._variance = standard_deviation ** 2
        self._a = (2 * np.pi * self._variance) ** -1.5


class GaussianSourceKCSD3D(GaussianSourceBase):
    _dtype = np.sqrt(0.5).__class__
    _fraction_of_erf_to_x_limit_in_0 = _dtype(2 / np.sqrt(np.pi))
    _x = _dtype(1.)
    _half = _dtype(0.5)
    _last = 2.
    _err = 1.
    while 0 < _err < _last:
        _radius_of_erf_to_x_limit_applicability = _x
        _last = _err
        _x *= _half
        _err = _fraction_of_erf_to_x_limit_in_0 - erf(_x) / _x

    def __init__(self, x, y, z, standard_deviation, conductivity):
        super(GaussianSourceKCSD3D, self).__init__(x, y, z, standard_deviation)
        self.conductivity = conductivity
        self._b = 0.25 / (np.pi * conductivity)
        self._c = np.sqrt(0.5) / standard_deviation

    def csd(self, X, Y, Z):
        return self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._variance)

    def potential(self, X, Y, Z):
        R = np.sqrt((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)
        Rc = R * self._c
        return self._b * np.where(Rc >= self._radius_of_erf_to_x_limit_applicability,
                                  erf(Rc) / R,
                                  self._c * self._fraction_of_erf_to_x_limit_in_0)


class SphericalSplineSourceBase(SourceBase):
    def __init__(self, x, y, z, nodes,
                 coefficients=((1,),
                               (-4, 12, -9, 2),
                               )):
        super(SphericalSplineSourceBase,
              self).__init__(x, y, z)
        self._nodes = nodes
        self._coefficients = coefficients
        self._a = 1.0 / self._integrate_spherically()

    def _integrate_spherically(self):
        acc = 0.0
        coeffs = [0, 0, 0]
        r0 = 0
        for r, coefficients in zip(self._nodes,
                                   self._coefficients):
            coeffs[3:] = [c / i
                          for i, c in enumerate(coefficients,
                                                start=3)]
            acc += (self._evaluate_polynomial(r, coeffs)
                    - self._evaluate_polynomial(r0, coeffs))
            r0 = r
        return 4 * np.pi * acc

    def csd(self, X, Y, Z):
        R = self._distance(X, Y, Z)
        CSD = np.zeros_like(R)
        r0 = 0
        for r, coefficients in zip(self._nodes,
                                   self._coefficients):
            IDX = (r0 <= R) & (R < r)
            CSD[IDX] = self._evaluate_polynomial(R[IDX],
                                                 coefficients)
            r0 = r

        return self._a * CSD

    def _distance(self, X, Y, Z):
        return np.sqrt(np.square(X - self.x)
                       + np.square(Y - self.y)
                       + np.square(Z - self.z))

    def _evaluate_polynomial(self, X, coefficients):
        ACC = 0
        for c in reversed(coefficients):
            ACC *= X
            ACC += c

        return ACC


class SphericalSplineSourceKCSD(SphericalSplineSourceBase):
    def __init__(self, x, y, z, nodes,
                 coefficients=((1,),
                               (-4, 12, -9, 2),
                               ),
                 conductivity=1):
        super(SphericalSplineSourceKCSD,
              self).__init__(x, y, z, nodes, coefficients)
        self.conductivity = conductivity

    def potential(self, X, Y, Z):
        R = self._distance(X, Y, Z)
        r0 = 0
        V = np.zeros_like(R)
        coefs_inside = [0, 0]
        coefs_outside = [0, 0, 0]
        for r, coefficients in zip(self._nodes,
                                   self._coefficients):
            coefs_inside[2:] = [c / i
                                for i, c in enumerate(coefficients,
                                                      start=2)]

            coefs_outside[3:] = [c / i
                                 for i, c in enumerate(coefficients,
                                                       start=3)]
            IDX = R <= r0  # inside both polynomial limits
            if IDX.any():
                V[IDX] += (self._evaluate_polynomial(r, coefs_inside)
                           - self._evaluate_polynomial(r0, coefs_inside))

            IDX = ~IDX & (R < r)  # within polynomial limits
            if IDX.any():
                # here is the bug
                _R = R[IDX]
                V[IDX] += (self._evaluate_polynomial(r, coefs_inside)
                           - self._evaluate_polynomial(_R, coefs_inside)
                           + (self._evaluate_polynomial(_R, coefs_outside)
                              - self._evaluate_polynomial(r0, coefs_outside)) / _R)

            IDX = R >= r  # outside both polynomial limits
            if IDX.any():
                _R = R[IDX]
                V[IDX] += (self._evaluate_polynomial(r, coefs_outside)
                           - self._evaluate_polynomial(r0, coefs_outside)) / _R

            r0 = r

        return V * self._a / self.conductivity


class PointSource(SourceBase):
    def __init__(self, x, y, z, conductivity, amplitude=1):
        super(PointSource, self).__init__(x, y, z)
        self.conductivity = conductivity
        self.a = amplitude * 0.25 / (np.pi * conductivity)

    def potential(self, X, Y, Z):
        return self.a / np.sqrt(np.square(X - self.x)
                                + np.square(Y - self.y)
                                + np.square(Z - self.z))


class InfiniteSliceSourceMOI(object):
    def __init__(self, x, y, z,
                 slice_thickness,
                 slice_conductivity, saline_conductivity, glass_conductivity=0,
                 amplitude=1,
                 n=128,
                 mask_invalid_space=True,
                 SourceClass=GaussianSourceKCSD3D,
                 **kwargs):
        self.x = x
        self.y = y
        self.z = z
        self.slice_thickness = slice_thickness
        self.n = n
        self.mask_invalid_space = mask_invalid_space
        self.amplitude = amplitude

        wtg = float(slice_conductivity - glass_conductivity) / (slice_conductivity + glass_conductivity)
        wts = float(slice_conductivity - saline_conductivity) / (slice_conductivity + saline_conductivity)
        self._source = SourceClass(x, y, z,
                                   conductivity=slice_conductivity,
                                   **kwargs)

        weights = [1]
        sources = [self._source]
        for i in range(n):
            weights.append(wtg**i * wts**(i+1))
            sources.append(SourceClass(x,
                                       y,
                                       2 * (i + 1) * slice_thickness - z,
                                       conductivity=slice_conductivity,
                                       **kwargs))
            weights.append(wtg**(i+1) * wts**i)
            sources.append(SourceClass(x,
                                       y,
                                       -2 * i * slice_thickness - z,
                                       conductivity=slice_conductivity,
                                       **kwargs))

        for i in range(1, n + 1):
            weights.append((wtg * wts)**i)
            sources.append(SourceClass(x,
                                       y,
                                       z + 2 * i * slice_thickness,
                                       conductivity=slice_conductivity,
                                       **kwargs))
            weights.append((wtg * wts)**i)
            sources.append(SourceClass(x,
                                       y,
                                       z - 2 * i * slice_thickness,
                                       conductivity=slice_conductivity,
                                       **kwargs))
        self._positive = [(w, s) for w, s in zip(weights, sources)
                          if w > 0]
        self._positive.sort(key=operator.itemgetter(0), reverse=False)
        self._negative = [(w, s) for w, s in zip(weights, sources)
                          if w < 0]
        self._negative.sort(key=operator.itemgetter(0), reverse=True)

    def _calculate_field(self, name, *args, **kwargs):
        return self.amplitude * (sum(w * getattr(s, name)(*args, **kwargs)
                                     for w, s in self._positive)
                                 + sum(w * getattr(s, name)(*args, **kwargs)
                                       for w, s in self._negative))

    def is_applicable(self, X, Y, Z):
        return (Z >= 0) & (Z <= self.slice_thickness)

    def _mask_invalid_space_if_requested(self, VALUE, MASK, fill_value=0):
        if self.mask_invalid_space:
            return np.where(MASK,
                            VALUE,
                            fill_value)
        return VALUE

    def csd(self, X, Y, Z):
        return self._mask_invalid_space_if_requested(
                        self._source.csd(X, Y, Z),
                        self.is_applicable(X, Y, Z))

    def actual_csd(self, X, Y, Z):
        return self._mask_invalid_space_if_requested(
                        self._calculate_field('csd', X, Y, Z),
                        self.is_applicable(X, Y, Z))

    def potential(self, X, Y, Z):
        return np.where(self.is_applicable(X, Y, Z),
                        self._calculate_field('potential', X, Y, Z),
                        np.nan)


class FourSphereModel(object):
    """
    Based on https://github.com/Neuroinflab/fourspheremodel
    by Chaitanya Chintaluri
    """
    Properties = collections.namedtuple('FourSpheres',
                                        ['brain',
                                        'csf',
                                        'skull',
                                        'scalp',
                                        ])

    I = 10.
    n = np.arange(1, 100)

    def __init__(self, conductivity, radius):
        self._set_radii(radius)
        self._set_conductivities(conductivity)

    def _set_radii(self, radius):
        self.radius = radius
        self.r12 = radius.brain / radius.csf
        self.r23 = radius.csf / radius.skull
        self.r34 = radius.skull / radius.scalp
        self.r21 = 1. / self.r12
        self.r32 = 1. / self.r23
        self.r43 = 1. / self.r34



    def _set_conductivities(self, conductivity):
        self.conductivity = conductivity
        self.s12 = conductivity.brain / conductivity.csf
        self.s23 = conductivity.csf / conductivity.skull
        self.s34 = conductivity.skull / conductivity.scalp

    def V(self, n):
        k = (n+1.) / n
        Factor = ((self.r34**n - self.r43**(n+1))
                  / (k * self.r34**n + self.r43**(n+1)))
        num = self.s34 / k - Factor
        den = self.s34 + Factor
        return num / den

    def Y(self, n):
        k = n / (n+1.)
        V_n = self.V(n)
        r23n = self.r23 ** n
        r32n1 = self.r32 ** (n + 1)
        Factor = ((r23n * k - V_n * r32n1)
                  / (r23n + V_n * r32n1))
        return (self.s23 * k - Factor) / (self.s23 + Factor)

    def Z(self, n):
        k = (n+1.) / n
        Y_n = self.Y(n)
        r12n = self.r12 ** n
        r21n1 = self.r21 ** (n + 1)
        return (r12n - k * Y_n * r21n1) / (r12n + Y_n * r21n1)

    def __call__(self, loc, P):
        return self._PointDipole(self, np.array(loc), P)

    class _PointDipole(object):
        def __init__(self, model, loc, P):
            self.model = model
            self.loc = loc
            self.P = np.reshape(P, (1, -1))
            self.dp_rad, self.dp_tan, r = self.decompose_dipole()
            self._set_dipole_r(r)

        def decompose_dipole(self):
            dist_dp = np.linalg.norm(self.loc)
            dp_rad = (np.dot(self.P, self.loc) / dist_dp) * (self.loc / dist_dp)
            dp_tan = self.P - dp_rad
            return dp_rad, dp_tan, dist_dp

        def _set_dipole_r(self, r):
            self.rz = r
            self.rz1 = r / self.model.radius.brain
            # self.r1z = 1. / self.rz1

        def __call__(self, X, Y, Z):
            # P = np.reshape(P, (1, -1))
            ELECTRODES = np.vstack([X, Y, Z]).T
            # dp_rad, dp_tan, r = self.decompose_dipole(P, dp_loc)
            # self._set_dipole_r(r)

            adjusted_theta = self.adjust_theta(self.loc, ELECTRODES)
            adjusted_phi_angle = self.adjust_phi_angle(self.dp_tan,
                                                       self.loc,
                                                       ELECTRODES)

            sign_rad = np.sign(np.dot(self.P, self.loc))
            mag_rad = sign_rad * np.linalg.norm(self.dp_rad)
            mag_tan = np.linalg.norm(self.dp_tan)  # sign_tan * np.linalg.norm(dp_tan)

            coef = self.H(self.model.n, self.model.radius.scalp)
            cos_theta = np.cos(adjusted_theta)

            # radial
            n_coef = self.model.n * coef
            rad_coef = np.insert(n_coef, 0, 0)
            Lprod = np.polynomial.legendre.Legendre(rad_coef)
            Lfactor_rad = Lprod(cos_theta)
            rad_phi = mag_rad * Lfactor_rad

            # #tangential
            Lfuncprod = [np.sum([C * lpmv(1, P_val, ct)
                                 for C, P_val in zip(coef, self.model.n)])
                         for ct in cos_theta]

            tan_phi = -1 * mag_tan * np.sin(adjusted_phi_angle) * np.array(Lfuncprod)
            return (rad_phi + tan_phi) / (4 * np.pi * self.model.conductivity.brain * (self.rz ** 2))

        def adjust_theta(self, dp_loc, ele_pos):
            ele_dist = np.linalg.norm(ele_pos, axis=1)
            dist_dp = np.linalg.norm(dp_loc)
            cos_theta = np.dot(ele_pos, dp_loc) / (ele_dist * dist_dp)
            if np.isnan(cos_theta).any():
                warnings.warn("invalid value of cos_theta", RuntimeWarning)
                cos_theta = np.nan_to_num(cos_theta)

            if (cos_theta > 1).any() or (cos_theta < -1).any():
                warnings.warn("cos_theta out of [-1, 1]", RuntimeWarning)
                cos_theta = np.maximum(-1, np.minimum(1, cos_theta))

            return np.arccos(cos_theta)

        def adjust_phi_angle(self, p, dp_loc, ele_pos):
            r_ele = np.sqrt(np.sum(ele_pos ** 2, axis=1))

            proj_rxyz_rz = (np.dot(ele_pos, dp_loc) / np.sum(dp_loc **2)).reshape(len(ele_pos),1) * dp_loc.reshape(1, 3)
            rxy = ele_pos - proj_rxyz_rz
            x = np.cross(p, dp_loc)
            cos_phi = np.dot(rxy, x.T) / np.dot(np.linalg.norm(rxy, axis=1).reshape(len(rxy), 1),
                                                np.linalg.norm(x, axis=1).reshape(1, len(x)))
            if abs(cos_phi).max() - 1 > 1e-10:
                warnings.warn("cos_phi out of [-1 - 1e-10, 1 + 1e-10]",
                              RuntimeWarning)

            if np.isnan(cos_phi).any():
                warnings.warn("invalid value of cos_phi", RuntimeWarning)
                cos_phi = np.nan_to_num(cos_phi)

            phi_temp = np.arccos(np.maximum(-1, np.minimum(1, cos_phi)))
            phi = phi_temp
            range_test = np.dot(rxy, p.T)
            for i in range(len(r_ele)):
                for j in range(len(p)):
                    if range_test[i, j] < 0:
                        phi[i,j] = 2 * np.pi - phi_temp[i, j]
            return phi.flatten()

        def H(self, n, r_ele):
            if r_ele < self.radius.brain:
                T1 = ((r_ele / self.radius.brain)**n) * self.A1(n)
                T2 = ((self.rz / r_ele)**(n + 1))
            elif r_ele < self.radius.csf:
                T1 = ((r_ele / self.radius.csf)**n) * self.A2(n)
                T2 = ((self.radius.csf / r_ele)**(n + 1)) * self.B2(n)
            elif r_ele < self.radius.skull:
                T1 = ((r_ele / self.radius.skull)**n) * self.A3(n)
                T2 = ((self.radius.skull / r_ele)**(n + 1)) * self.B3(n)
            elif r_ele <= self.radius.scalp:
                T1 = ((r_ele / self.radius.scalp)**n) * self.A4(n)
                T2 = ((self.radius.scalp / r_ele)**(n + 1)) * self.B4(n)
            else:
                print("Invalid electrode position")
                return
            return T1 + T2

        def A1(self, n):
            Z_n = self.Z(n)
            k = (n + 1.) / n
            return self.rz1 ** (n + 1) * (Z_n + self.s12 * k) / (self.s12 - Z_n)

        def A2(self, n):
            return ((self.A1(n) + self.rz1 ** (n + 1))
                    / (self.Y(n) * self.r21 ** (n + 1) + self.r12 ** n))

        def A3(self, n):
            return ((self.A2(n) + self.B2(n))
                    / (self.r23 ** n + self.V(n) * self.r32 ** (n + 1)))

        def B2(self, n):
            return self.A2(n) * self.Y(n)

        def A4(self, n):
            k = (n+1.) / n
            return k * ((self.A3(n) + self.B3(n))
                        / (k * self.r34 ** n + self.r43 ** (n + 1)))

        def B3(self, n):
            return self.A3(n) * self.V(n)

        def B4(self, n):
            return self.A4(n) * n / (n + 1.)

        def __getattr__(self, name):
            return getattr(self.model, name)


def cv(reconstructor, measured, regularization_parameters):
    errors = []

    for regularization_parameter in regularization_parameters:
        logger.info('cv(): error estimation for regularization parameter: {:g}'.format(regularization_parameter))
        ERR = reconstructor.leave_one_out_errors(measured,
                                                 regularization_parameter)
        errors.append(np.sqrt(np.square(ERR).mean()))

    return errors


def altitude_azimuth_mesh(base, step, alternate=True):
    for i, altitude in enumerate(np.linspace(base,
                                             np.pi / 2,
                                             int(round((np.pi / 2 - base) / step) + 1))):
        for azimuth in (np.linspace(-1, 1,
                                    int(round(np.cos(altitude) * 2 * np.pi / step)) + 1,
                                    endpoint=False)
                        + (0.5 * (-1) ** i if alternate else 1)) * np.pi:
            yield altitude, azimuth


if __name__ == '__main__':
    import common
    import pandas as pd

    BRAIN_CONDUCTIVITY = 1. / 300.  # S / cm
    CONDUCTIVITY = common.FourSphereModel.Properties(1.00 * BRAIN_CONDUCTIVITY,
                                                     5.00 * BRAIN_CONDUCTIVITY,
                                                     0.05 * BRAIN_CONDUCTIVITY,
                                                     1.00 * BRAIN_CONDUCTIVITY)
    RADIUS = common.FourSphereModel.Properties(7.9, 8.0, 8.5, 9.0)
    X, Y, Z = np.meshgrid(np.linspace(-8.9, 8.9, 10),
                          np.linspace(-8.9, 8.9, 10),
                          np.linspace(-8.9, 8.9, 10))
    ELECTRODES = pd.DataFrame({'X': X.flatten(),
                               'Y': Y.flatten(),
                               'Z': Z.flatten(),
                               })
    DF = ELECTRODES.copy()
    oldFourSM = common.FourSphereModel(CONDUCTIVITY,
                                       RADIUS,
                                       ELECTRODES)
    newFourSM = FourSphereModel(CONDUCTIVITY,
                                RADIUS)

    src_pos = np.array([-0.05, 7.835, -0.0353])
    snk_pos = np.array([0.00, 7.765, 0.0353])
    DF['OLD'] = oldFourSM.compute_phi(src_pos, snk_pos)
    P = oldFourSM.I * (src_pos - snk_pos)
    LOC = 0.5 * (src_pos + snk_pos)
    newDipoleFourSM = newFourSM(list(LOC), list(P))
    DF['NEW'] = newDipoleFourSM(ELECTRODES.X,
                                ELECTRODES.Y,
                                ELECTRODES.Z)
    assert np.abs((DF.OLD - DF.NEW) / DF.OLD).max() < 1e-10

    dipole = newFourSM([7.437628862425826, 1.9929066472894097, -1.3662702569423635e-15],
                       [0.0, 0.0, 1.0])
    assert not np.isnan(dipole(7.211087502867844, 5.368455739408048, -1.3246552843137878e-15))

    dipole = newFourSM([7.437628862425826, 1.9929066472894097, -1.3662702569423635e-15],
                       [0.0, 0.0, -1.0])
    assert not np.isnan(dipole(7.211087502867844, 5.368455739408048, -1.3246552843137878e-15))


    import scipy.stats as ss
    np.random.seed(42)
    for standard_deviation in [0.1, 1, 10]:
        for MEAN in np.random.normal(size=(100, 3)):
            X = np.random.normal(size=(100, 3))
            EXPECTED = ss.multivariate_normal.pdf(X,
                                                  mean=MEAN,
                                                  cov=np.diag(np.full(3,
                                                                      standard_deviation**2)))
            gauss_function = Gauss3D(MEAN[0], MEAN[1], MEAN[2], standard_deviation)
            OBSERVED = gauss_function(*X.T)
            max_error = abs(EXPECTED - OBSERVED).max()
            assert max_error < 150 * np.finfo(float).eps