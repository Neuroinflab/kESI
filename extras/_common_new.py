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

    def __init__(self, conductivity, radius, n=100):
        self.n = np.arange(1, n)
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
        try:
            return self._V
        except AttributeError:
            n = self.n

        k = (n+1.) / n
        Factor = ((self.r34**n - self.r43**(n+1))
                  / (k * self.r34**n + self.r43**(n+1)))
        num = self.s34 / k - Factor
        den = self.s34 + Factor
        self._V = num / den
        return self._V

    def Y(self, n):
        try:
            return self._Y
        except AttributeError:
            n = self.n

        k = n / (n+1.)
        V_n = self.V(n)
        r23n = self.r23 ** n
        r32n1 = self.r32 ** (n + 1)
        Factor = ((r23n * k - V_n * r32n1)
                  / (r23n + V_n * r32n1))
        self._Y = (self.s23 * k - Factor) / (self.s23 + Factor)
        return self._Y

    def Z(self, n):
        try:
            return self._Z
        except AttributeError:
            n = self.n

        k = (n+1.) / n
        Y_n = self.Y(n)
        r12n = self.r12 ** n
        r21n1 = self.r21 ** (n + 1)
        self._Z = (r12n - k * Y_n * r21n1) / (r12n + Y_n * r21n1)
        return self._Z

    def __call__(self, loc, P):
        return self._PointDipole(self, np.array(loc), P)

    class _PointDipole(object):
        def __init__(self, model, dipole_loc, dipole_moment):
            self.model = model
            self.set_dipole_loc(dipole_loc)
            self.decompose_dipole(np.reshape(dipole_moment,
                                             (1, -1)))
            self._set_dipole_r()

        def set_dipole_loc(self, loc):
            self.loc_r = np.sqrt(np.square(loc).sum())
            self.loc_v = (np.reshape(loc, (1, -1)) / self.loc_r
                          if self.loc_r != 0
                          else np.array([[0, 0, 1]]))

        @property
        def loc(self):
            return self.loc_r * self.loc_v.flatten()

        @property
        def rz(self):
            return self.loc_r

        def decompose_dipole(self, P):
            self.p_rad = self.north_vector(P)
            self.p_tan = P - self.p_rad

        def north_vector(self, V):
            return np.matmul(self.north_projection(V),
                             self.loc_v)

        def north_projection(self, V):
            return np.matmul(V,
                             self.loc_v.T)

        def _set_dipole_r(self):
            self.rz1 = self.loc_r / self.model.radius.brain

        def __call__(self, X, Y, Z):
            ELECTRODES = np.vstack([X, Y, Z]).T

            ele_dist, adjusted_theta = self.adjust_theta(self.loc, ELECTRODES)
            tan_cosinus = self.tan_versor_cosinus(ELECTRODES).flatten()

            sign_rad = np.sign(self.north_projection(self.p_rad))
            mag_rad = sign_rad * np.linalg.norm(self.p_rad)
            mag_tan = np.linalg.norm(self.p_tan)  # sign_tan * np.linalg.norm(dp_tan)

            potentials = np.zeros_like(tan_cosinus)
            for i, (_r, _theta, _cos) in enumerate(zip(ele_dist,
                                                   adjusted_theta,
                                                   tan_cosinus)):
                try:
                    # coef = self.H(self.n, _r)
                    coef = self.H(_r)

                    cos_theta = np.cos(_theta)

                    # radial
                    n_coef = self.n * coef
                    rad_coef = np.insert(n_coef, 0, 0)
                    Lprod = np.polynomial.legendre.Legendre(rad_coef)
                    Lfactor_rad = Lprod(cos_theta)
                    rad_phi = mag_rad * Lfactor_rad

                    # #tangential
                    Lfuncprod = np.sum([C * lpmv(1, P_val, cos_theta)
                                        for C, P_val in zip(coef, self.n)])

                    tan_phi = -1 * mag_tan * _cos * np.array(Lfuncprod)
                    potentials[i] = rad_phi + tan_phi
                except:
                    potentials[i] = np.nan

            return potentials / (4 * np.pi * self.model.conductivity.brain * (self.rz ** 2))

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

            return ele_dist, np.arccos(cos_theta)

        def tan_versor_cosinus(self, ele_pos):
            ele_north = self.north_vector(ele_pos)
            ele_parallel = ele_pos - ele_north
            ele_parallel_v = ele_parallel / np.sqrt(np.square(ele_parallel).sum(axis=1).reshape(-1, 1))

            tan_parallel = self.p_tan - self.north_vector(self.p_tan)
            tan_r = np.sqrt(np.square(tan_parallel).sum())
            if tan_r == 0:
                warnings.warn("no tangential dipole",
                              RuntimeWarning)
                return np.zeros((ele_pos.shape[0], 1))

            tan_parallel_v = tan_parallel / tan_r
            cos = np.matmul(ele_parallel_v,
                            tan_parallel_v.T)

            if abs(cos).max() - 1 > 1e-10:
                warnings.warn("cos out of [-1 - 1e-10, 1 + 1e-10]",
                              RuntimeWarning)

            if np.isnan(cos).any():
                warnings.warn("invalid value of cos", RuntimeWarning)
                cos = np.nan_to_num(cos)

            return cos

        # def H(self, n, r_ele):
        def H(self, r_ele):
            n = self.n
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

        @property
        def n(self):
            return self.model.n

        def A1(self, n):
            try:
                return self._A1
            except AttributeError:
                n = self.n

            Z_n = self.Z(n)
            k = (n + 1.) / n
            self._A1 = self.rz1 ** (n + 1) * (Z_n + self.s12 * k) / (self.s12 - Z_n)
            return self._A1

        def A2(self, n):
            try:
                return self._A2
            except AttributeError:
                n = self.n

            self._A2 = ((self.A1(n) + self.rz1 ** (n + 1))
                    / (self.Y(n) * self.r21 ** (n + 1) + self.r12 ** n))

            return self._A2

        def A3(self, n):
            try:
                return self._A3
            except AttributeError:
                n = self.n

            self._A3 = ((self.A2(n) + self.B2(n))
                    / (self.r23 ** n + self.V(n) * self.r32 ** (n + 1)))
            return self._A3

        def B2(self, n):
            try:
                return self._B2
            except AttributeError:
                n = self.n

            self._B2 = self.A2(n) * self.Y(n)
            return self._B2

        def A4(self, n):
            try:
                return self._A4
            except AttributeError:
                n = self.n

            k = (n+1.) / n
            self._A4 = k * ((self.A3(n) + self.B3(n))
                        / (k * self.r34 ** n + self.r43 ** (n + 1)))
            return self._A4

        def B3(self, n):
            try:
                return self._B3
            except AttributeError:
                n = self.n

            self._B3 = self.A3(n) * self.V(n)
            return self._B3

        def B4(self, n):
            try:
                return self._B4
            except AttributeError:
                n = self.n

            self._B4 = self.A4(n) * n / (n + 1.)
            return self._B4

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
    SCALP_R = 9.0
    RADIUS = common.FourSphereModel.Properties(7.9, 8.0, 8.5, SCALP_R)

    N = 1000
    np.random.seed(42)
    ELECTRODES = pd.DataFrame({
        'PHI': np.random.uniform(-np.pi, np.pi, N),
        'THETA': 2 * np.arcsin(np.sqrt(np.random.uniform(0, 1, N))) - np.pi / 2
        })

    ELECTRODES['X'] = SCALP_R * np.sin(ELECTRODES.THETA)
    _XY_R = SCALP_R * np.cos(ELECTRODES.THETA)
    ELECTRODES['Y'] = _XY_R * np.cos(ELECTRODES.PHI)
    ELECTRODES['Z'] = _XY_R * np.sin(ELECTRODES.PHI)

    DF = ELECTRODES.copy()
    oldFourSM = common.FourSphereModel(CONDUCTIVITY,
                                       RADIUS,
                                       ELECTRODES[['X', 'Y', 'Z']])
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
                                ELECTRODES.Z).flatten()
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