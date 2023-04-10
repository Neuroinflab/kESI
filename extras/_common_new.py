#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2021 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Institute of Applied Psychology;  #
#    Faculty of Management and Social Communication; Jagiellonian University) #
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
import configparser
import json

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
    @classmethod
    def fromJSON(cls, _file, **kwargs):
        _json = json.load(_file)
        _json.update(kwargs)
        return cls(**_json)

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

    def __init__(self, x, y, z, standard_deviation, conductivity=1.0):
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
        self._csd_polynomials = coefficients
        self._a = 1.0 / self._integrate_spherically()

    def _integrate_spherically(self):
        acc = 0.0
        coeffs = [0, 0, 0]
        r0 = 0
        for r, coefficients in zip(self._nodes,
                                   self._csd_polynomials):
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
                                   self._csd_polynomials):
            IDX = (r0 <= R) & (R < r)
            CSD[IDX] = self._evaluate_polynomial(R[IDX],
                                                 coefficients)
            r0 = r

        return self._a * CSD

    def _distance(self, X, Y, Z):
        return np.sqrt(np.square(X - self.x)
                       + np.square(Y - self.y)
                       + np.square(Z - self.z))

    @staticmethod
    def _evaluate_polynomial(X, coefficients):
        # NOTE: benchmark before reimplementation with dedicated NumPy solution
        # For NumPy v. 1.21.6 Python v. 3.7.12 the method is twice as fast
        # as either `np.polyval()` function or object
        # of `np.polynomial.Polynomial` class.
        ACC = 0
        for c in reversed(coefficients):
            ACC *= X
            ACC += c

        return ACC

    @property
    def radius(self):
        return self._nodes[-1]

    def toJSON(self, file):
        json.dump(self._constructor_args(),
                  file,
                  indent=2)

    def _constructor_args(self):
        return {"x": self.x,
                "y": self.y,
                "z": self.z,
                "nodes": self._nodes,
                "coefficients": self._csd_polynomials}


class SphericalSplineSourceKCSD(SphericalSplineSourceBase):
    """
    Notes
    -----

    Potentials in a medium of constant, scalar conductivity are calculated by
    integrating :math:`\int V(R)` [1]_, where the surface charge density
    :math:`\sigma_0` is substituted by surface CSD :math:`CSD(R) dR`,
    and the vacuum permittivity :math:`\vareps_0`, by medium conductivity
    :math:`\sigma`, thus transforming the electrostatic problem into electric.

    References
    ----------

    .. [1] Markus Zahn, **Pole elektromagnetyczne**,
       1989 Warszawa, Państwowe Wydawnictwo Naukowe, ISBN: 83-01-07693-3,
       p. 101, eq. 21
       (original title: Electromagnetic Field Theory: a problem solving approach)
    """
    def __init__(self, x, y, z, nodes,
                 coefficients=((1,),
                               (-4, 12, -9, 2),
                               ),
                 conductivity=1.0):
        super(SphericalSplineSourceKCSD,
              self).__init__(x, y, z, nodes, coefficients)

        self._calculate_potential_coefficients()
        self.conductivity = conductivity

    def _calculate_potential_coefficients(self):
        self._offsetted_external_shell_potential_polynomials = list(map(
                                       self._offsetted_external_shell_potential,
                                       self._csd_polynomials))
        self._internal_sphere_potential_dividend_polynomials = list(map(
                                       self._internal_sphere_potential_dividend,
                                       self._csd_polynomials))

    @staticmethod
    def _offsetted_external_shell_potential(csd):
        """
        Calculate a potential generated at an electrode by a CSD shell external
        to the electrode.

        Parameters
        ----------
        csd : [float, ...]
            Coefficients of a CSD-defining polynomial in increasing order.

        Returns
        -------
        V : [float, ...]
            Coefficients of polynomial `V` such that `V(R) - V(r0)`
            is a potential at radius `r0` generated by an external shell
            (spanning from `r0` to `R`) of CSD given by `csd`.

        Notes
        -----
        The :math:`CSD(r)` function may be defined by `csd` as:

        >>> CSD = lambda r: sum(a * r ** i for i, a in enumerate(csd))
        """
        return [0.0] * 2 + [a / i for i, a in enumerate(csd, start=2)]

    @staticmethod
    def _internal_sphere_potential_dividend(csd):
        """
        Calculate a dividend of a potential generated at an electrode by a CSD
        sphere internal to the electrode.

        Parameters
        ----------
        csd : [float, ...]
            Coefficients of a CSD-defining polynomial in increasing order.

        Returns
        -------
        V : [float, ...]
            Coefficients of polynomial `V` such that `V(R) / r0`
            is a potential at radius `r0` generated by an internal sphere
            (of radius `R < r0`) of CSD given by `csd`.

        Notes
        -----
        The :math:`CSD(r)` function may be defined by `csd` as:

        >>> CSD = lambda r: sum(a * r ** i for i, a in enumerate(csd))
        """
        return [0.0] * 3 + [a / i for i, a in enumerate(csd, start=3)]

    def potential(self, X, Y, Z):
        R = self._distance(X, Y, Z)
        r0 = 0
        V = np.zeros_like(R)

        for r, p_ext, p_int in zip(
                          self._nodes,
                          self._offsetted_external_shell_potential_polynomials,
                          self._internal_sphere_potential_dividend_polynomials):
            IDX = R <= r0  # inside both polynomial limits
            if IDX.any():
                V[IDX] += self._external_shell_potential(p_ext, r0, r)

            IDX = ~IDX & (R < r)  # within polynomial limits
            if IDX.any():
                # here is the bug  # 2023-04-10: the comment seems to be rotten
                _R = R[IDX]
                V[IDX] += (self._external_shell_potential(p_ext, _R, r)
                           + self._internal_shell_potential(p_int, _R, r0, _R))

            IDX = R >= r  # outside both polynomial limits
            if IDX.any():
                _R = R[IDX]
                V[IDX] += self._internal_shell_potential(p_int, r, r0, _R)

            r0 = r

        return V * self._a / self.conductivity

    def _internal_shell_potential(self,
                                  p_internal,
                                  r_external,
                                  r_internal,
                                  r_electrode):
        return (self._evaluate_polynomial(r_external, p_internal)
                - self._evaluate_polynomial(r_internal, p_internal)) / r_electrode

    def _external_shell_potential(self, p_csd, r_internal, r_external):
        return (self._evaluate_polynomial(r_external, p_csd)
                - self._evaluate_polynomial(r_internal, p_csd))

    def _constructor_args(self):
        d = super()._constructor_args()
        d['conductivity'] = self.conductivity
        return d


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
    _LAYERS = ['brain',
               'csf',
               'skull',
               'scalp',
               ]
    class Properties(collections.namedtuple('FourSpheres',
                                            _LAYERS)):
        @classmethod
        def from_config(cls, path, field):
            config = configparser.ConfigParser()
            config.read(path)
            return cls(*[config.getfloat(section, field)
                         for section in FourSphereModel._LAYERS])

    def __init__(self, conductivity, radius, n=100):
        self.n = np.arange(1, n)
        self._set_radii(radius)
        self._set_conductivities(conductivity)

    @classmethod
    def from_config(cls, path, n=100):
        return cls(cls.Properties.from_config(path, 'conductivity'),
                   cls.Properties.from_config(path, 'radius'),
                   n)

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
        return self._PointDipole(self,
                                 np.reshape(loc,
                                            (1, 3)),
                                 np.reshape(P,
                                            (1, 3)))

    class _PointDipole(object):
        def __init__(self, model, dipole_loc, dipole_moment):
            self.model = model
            self.set_dipole_loc(dipole_loc)
            self.decompose_dipole(dipole_moment)
            self._set_dipole_r()

        def set_dipole_loc(self, loc):
            self.loc_r = np.sqrt(np.square(loc).sum())
            self.loc_v = (loc / self.loc_r
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

            ele_dist = np.linalg.norm(ELECTRODES, axis=1)
            COS_THETA = self.cos_theta(ELECTRODES / ele_dist.reshape(-1, 1))
            tan_cosinus = self.tan_versor_cosinus(ELECTRODES).flatten()

            COEF = self.H_v(ele_dist)
            LPMV = lpmv(1,                      # expensive for n >= 10_000;
                        self.n.reshape(1, -1),  # line_profiler claims 99.7%
                        COS_THETA)              # experimental complexity O(n^2)
            LFUNCPROD = (COEF * LPMV).sum(axis=1)

            NCOEF = self.n * COEF
            RAD_COEF = np.hstack([np.zeros((COEF.shape[0], 1)),
                                  NCOEF])
            LFACTOR = np.polynomial.legendre.legval(COS_THETA.flatten(),
                                                    RAD_COEF.T,
                                                    tensor=False)

            sign_rad = np.sign(self.north_projection(self.p_rad))
            mag_rad = sign_rad * np.linalg.norm(self.p_rad)
            mag_tan = np.linalg.norm(self.p_tan)  # sign_tan * np.linalg.norm(dp_tan)

            tan_potential = -mag_tan * tan_cosinus * LFUNCPROD
            rad_potential = mag_rad * LFACTOR
            potentials = tan_potential + rad_potential
            return potentials / (4 * np.pi * self.model.conductivity.brain * (self.rz ** 2))

        def cos_theta(self, ele_versors):
            cos_theta = self.north_projection(ele_versors)

            if np.isnan(cos_theta).any():
                warnings.warn("invalid value of cos_theta", RuntimeWarning)
                cos_theta = np.nan_to_num(cos_theta)

            if (cos_theta > 1).any() or (cos_theta < -1).any():
                warnings.warn("cos_theta out of [-1, 1]", RuntimeWarning)
                cos_theta = np.maximum(-1, np.minimum(1, cos_theta))

            return cos_theta

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


        def H_v(self, r_ele):
            COEF = np.full((len(r_ele), len(self.n)),
                           np.nan)
            IDX_LOW = r_ele >= self.loc_r

            for i, r in zip(np.arange(len(r_ele))[~IDX_LOW],
                            r_ele[~IDX_LOW]):
                print("Invalid position of electrode #{:d}: {:f} (off by {:e})".format(
                      i, r, r - self.loc_r))

            IDX_HIGH = r_ele < self.radius.brain
            IDX = IDX_LOW & IDX_HIGH
            if IDX.any():
                _r_ele = r_ele[IDX].reshape(-1, 1)
                T1 = ((_r_ele / self.radius.brain) ** self.n) * self.A1()
                T2 = ((self.rz / _r_ele) ** (self.n + 1))
                COEF[IDX, :] = T1 + T2

            IDX_LOW[IDX_HIGH] = False
            IDX_HIGH = r_ele < self.radius.csf
            IDX = IDX_LOW & IDX_HIGH
            if IDX.any():
                _r_ele = r_ele[IDX].reshape(-1, 1)
                T1 = ((_r_ele / self.radius.csf) ** self.n) * self.A2()
                T2 = ((self.radius.csf / _r_ele) ** (self.n + 1)) * self.B2()
                COEF[IDX, :] = T1 + T2

            IDX_LOW[IDX_HIGH] = False
            IDX_HIGH = r_ele < self.radius.skull
            IDX = IDX_LOW & IDX_HIGH
            if IDX.any():
                _r_ele = r_ele[IDX].reshape(-1, 1)
                T1 = ((_r_ele / self.radius.skull) ** self.n) * self.A3()
                T2 = ((self.radius.skull / _r_ele) ** (self.n + 1)) * self.B3()
                COEF[IDX, :] = T1 + T2

            IDX_LOW[IDX_HIGH] = False
            IDX_HIGH = r_ele <= self.radius.scalp
            IDX = IDX_LOW & IDX_HIGH
            if IDX.any():
                _r_ele = r_ele[IDX].reshape(-1, 1)
                T1 = ((_r_ele / self.radius.scalp) ** self.n) * self.A4()
                T2 = ((self.radius.scalp / _r_ele) ** (self.n + 1)) * self.B4()
                COEF[IDX, :] = T1 + T2

            for i, r in zip(np.arange(len(r_ele))[~IDX_HIGH],
                            r_ele[~IDX_HIGH]):
                print("Invalid position of electrode #{:d}: {:f} (off by {:e})".format(
                      i, r, r - self.radius.scalp))

            return COEF

        @property
        def n(self):
            return self.model.n

        def A1(self, n=None):
            try:
                return self._A1
            except AttributeError:
                n = self.n

            Z_n = self.Z(n)
            k = (n + 1.) / n
            self._A1 = self.rz1 ** (n + 1) * (Z_n + self.s12 * k) / (self.s12 - Z_n)
            return self._A1

        def A2(self, n=None):
            try:
                return self._A2
            except AttributeError:
                n = self.n

            self._A2 = ((self.A1(n) + self.rz1 ** (n + 1))
                    / (self.Y(n) * self.r21 ** (n + 1) + self.r12 ** n))

            return self._A2

        def A3(self, n=None):
            try:
                return self._A3
            except AttributeError:
                n = self.n

            self._A3 = ((self.A2(n) + self.B2(n))
                    / (self.r23 ** n + self.V(n) * self.r32 ** (n + 1)))
            return self._A3

        def B2(self, n=None):
            try:
                return self._B2
            except AttributeError:
                n = self.n

            self._B2 = self.A2(n) * self.Y(n)
            return self._B2

        def A4(self, n=None):
            try:
                return self._A4
            except AttributeError:
                n = self.n

            k = (n+1.) / n
            self._A4 = k * ((self.A3(n) + self.B3(n))
                        / (k * self.r34 ** n + self.r43 ** (n + 1)))
            return self._A4

        def B3(self, n=None):
            try:
                return self._B3
            except AttributeError:
                n = self.n

            self._B3 = self.A3(n) * self.V(n)
            return self._B3

        def B4(self, n=None):
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


def one_hot_vector(length, hot_position, hot=1, cold=0):
    return np.where(np.arange(length) == hot_position, hot, cold)


def shape(dimensions, axis):
    return one_hot_vector(dimensions, axis, hot=-1, cold=1)


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


    # TESTS fromJSON
    from io import StringIO

    class TestFromJSON(SourceBase):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kwargs = {'a': 42, 'b': 1337}
    src = TestFromJSON.fromJSON(StringIO(json.dumps(kwargs)))
    assert isinstance(src, TestFromJSON)
    assert kwargs == src.kwargs

    expected = kwargs.copy()
    expected.update(a=0, c='a')
    src = TestFromJSON.fromJSON(StringIO(json.dumps(kwargs)),
                                a=0,
                                c='a')
    assert expected == src.kwargs

    # TEST SphericalSplineSourceBase.toJSON()
    _x, _y, _z = 1, 2, 3
    _nodes = [1, 2]
    expected = SphericalSplineSourceBase(x=_x, y=_y, z=_z,
                                         nodes=_nodes,
                                         coefficients=[[1], [2, -1]])
    _file = StringIO()
    expected.toJSON(_file)
    _file.seek(0)
    observed = SphericalSplineSourceBase.fromJSON(_file)
    assert observed.x == _x
    assert observed.y == _y
    assert observed.z == _z
    assert observed._nodes == _nodes
    assert observed.csd(1, 2, 1.5) == expected.csd(1, 2, 1.5)


    # TEST SphericalSplineSourceBase.radius
    assert SphericalSplineSourceBase(0, 0, 0, [2], [[1]]).radius == 2
    assert SphericalSplineSourceBase(0, 0, 0, [1, 2], [[1]] * 2).radius == 2


    # TESTS one_hot_vector()
    assert np.all(one_hot_vector(1, 0) == [1])
    assert np.all(one_hot_vector(2, 0) == [1, 0])
    assert np.all(one_hot_vector(2, 1) == [0, 1])
    assert np.all(one_hot_vector(2, 1, hot=-1) == [0, -1])
    assert np.all(one_hot_vector(2, 1, hot=-1, cold=1) == [1, -1])