#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
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
import collections
import logging
import operator

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import lpmv, erf


logger = logging.getLogger(__name__)


class FourSphereModel(object):
    """
    Based on https://github.com/Neuroinflab/fourspheremodel
    by Chaitanya Chintaluri
    """
    Properies = collections.namedtuple('FourSpheres',
                                       ['brain',
                                        'csf',
                                        'skull',
                                        'scalp',
                                        ])

    I = 10.
    n = np.arange(1, 100)

    def __init__(self, conductivity, radius, ELECTRODES):
        self.ELECTRODES = ELECTRODES
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

    def _set_dipole_r(self, r):
        self.rz = r
        self.rz1 = r / self.radius.brain
        self.r1z = 1. / self.rz1

    def _set_conductivities(self, conductivity):
        self.conductivity = conductivity
        self.s12 = conductivity.brain / conductivity.csf
        self.s23 = conductivity.csf / conductivity.skull
        self.s34 = conductivity.skull / conductivity.scalp

    def V(self, n):
        k = (n+1.) / n
        Factor = ( (self.r34**n - (self.r43**(n+1)) ) / ( (k*(self.r34**n)) + (self.r43**(n+1)) ) )
        num = (self.s34/k) - Factor
        den = self.s34 + Factor
        return num / den


    def Y(self, n):
        k = n / (n+1.)
        Factor = ( ( (self.r23**n) * k) - self.V(n)*(self.r32**(n+1))) / (self.r23**n + self.V(n)*(self.r32**(n+1)))
        num = (self.s23*k) - Factor
        den = self.s23 + Factor
        return num / den


    def Z(self, n):
        k = (n+1.) / n
        num = (self.r12**n - k*self.Y(n)*(self.r21**(n+1)) ) / (self.r12**n + self.Y(n)*(self.r21**(n+1)))
        return num


    def A1(self, n):
        num = (self.rz1**(n+1))* (self.Z(n) + self.s12*((n+1.)/n))
        den = self.s12 - self.Z(n)
        return num / den


    def A2(self, n):
        num = self.A1(n) + (self.rz1**(n+1))
        den = (self.Y(n)*(self.r21**(n+1))) + self.r12**n
        return num / den


    def B2(self, n):
        return self.A2(n)*self.Y(n)


    def A3(self, n):
        num = self.A2(n) + self.B2(n)
        den = self.r23**n + (self.V(n)*(self.r32**(n+1)))
        return num / den


    def B3(self, n):
        return self.A3(n)*self.V(n)


    def A4(self, n):
        num = self.A3(n) + self.B3(n)
        k = (n+1.) / n
        den = (k*(self.r34**n)) + (self.r43**(n+1))
        return k*(num / den)


    def B4(self, n):
        return self.A4(n)* (n / (n+1.))


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


    def adjust_theta(self, src_pos, snk_pos):
        ele_pos = self.ELECTRODES.values
        dp_loc = (np.array(src_pos) + np.array(snk_pos)) / 2.
        ele_dist = np.linalg.norm(ele_pos, axis=1)
        dist_dp = np.linalg.norm(dp_loc)
        cos_theta = np.dot(ele_pos, dp_loc) / (ele_dist * dist_dp)
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        return theta


    def adjust_phi_angle(self, p, src_pos, snk_pos):
        ele_pos = self.ELECTRODES.values
        r_ele = np.sqrt(np.sum(ele_pos ** 2, axis=1))
        dp_loc = (np.array(src_pos) + np.array(snk_pos)) / 2.
        proj_rxyz_rz = (np.dot(ele_pos, dp_loc) / np.sum(dp_loc **2)).reshape(len(ele_pos),1) * dp_loc.reshape(1, 3)
        rxy = ele_pos - proj_rxyz_rz
        x = np.cross(p, dp_loc)
        cos_phi = np.dot(rxy, x.T) / np.dot(np.linalg.norm(rxy, axis=1).reshape(len(rxy),1), np.linalg.norm(x, axis=1).reshape(1, len(x)))
        cos_phi = np.nan_to_num(cos_phi)
        phi_temp = np.arccos(cos_phi)
        phi = phi_temp
        range_test = np.dot(rxy, p.T)
        for i in range(len(r_ele)):
            for j in range(len(p)):
                if range_test[i, j] < 0:
                    phi[i,j] = 2 * np.pi - phi_temp[i, j]
        return phi.flatten()


    def decompose_dipole(self, I, src_pos, snk_pos):
        P, dp_loc = self.get_dipole_moment_and_loc(I,
                                                   np.array(src_pos),
                                                   np.array(snk_pos))
        P = P.reshape((1, -1))

        dist_dp = np.linalg.norm(dp_loc)
        dp_rad = (np.dot(P, dp_loc) / dist_dp) * (dp_loc / dist_dp)
        dp_tan = P - dp_rad
        return P, dp_rad, dp_tan, dist_dp

    def get_dipole_moment_and_loc(self, I, SRC, SNK):
        return (I * (SRC - SNK)), (0.5 * (SRC + SNK))

    def compute_phi(self, src_pos, snk_pos):
        P, dp_rad, dp_tan, r = self.decompose_dipole(self.I, src_pos, snk_pos)
        self._set_dipole_r(r)

        adjusted_theta = self.adjust_theta(src_pos, snk_pos)

        adjusted_phi_angle = self.adjust_phi_angle(dp_tan, src_pos, snk_pos)  # params.phi_angle_r

        dp_loc = (np.array(src_pos) + np.array(snk_pos)) / 2
        sign_rad = np.sign(np.dot(P, dp_loc))
        mag_rad = sign_rad * np.linalg.norm(dp_rad)
        mag_tan = np.linalg.norm(dp_tan)  # sign_tan * np.linalg.norm(dp_tan)

        coef = self.H(self.n, self.radius.scalp)
        cos_theta = np.cos(adjusted_theta)

        # radial
        n_coef = self.n * coef
        rad_coef = np.insert(n_coef, 0, 0)
        Lprod = np.polynomial.legendre.Legendre(rad_coef)
        Lfactor_rad = Lprod(cos_theta)
        rad_phi = mag_rad * Lfactor_rad

        # #tangential
        Lfuncprod = []
        for tt in range(len(self.ELECTRODES)):
            Lfuncprod.append(np.sum([C * lpmv(1, P_val, cos_theta[tt])
                                     for C, P_val in zip(coef, self.n)]))
        tan_phi = -1 * mag_tan * np.sin(adjusted_phi_angle) * np.array(Lfuncprod)

        return (rad_phi + tan_phi) / (4 * np.pi * self.conductivity.brain * (self.rz ** 2))


class PolarBase(object):
    def __init__(self, ROW):
        y = ROW.R * np.sin(ROW.ALTITUDE)
        r = ROW.R * np.cos(ROW.ALTITUDE)
        x = r * np.sin(ROW.AZIMUTH)
        z = r * np.cos(ROW.AZIMUTH)
        self.init(x, y, z, ROW)


class ElectrodeAware(object):
    def __init__(self, ELECTRODES, *args, **kwargs):
        super(ElectrodeAware, self).__init__(*args, **kwargs)
        self._ELECTRODES = ELECTRODES

    def potential(self, electrodes):
        return super(ElectrodeAware, self).potential(self._ELECTRODES.loc[electrodes])


class CartesianBase(object):
    def __init__(self, ROW):
        self.init(ROW.X, ROW.Y, ROW.Z, ROW)


class GaussianSourceBase(object):
    def init(self, x, y, z, ROW):
        self.x = x
        self.y = y
        self.z = z
        self._sigma2 = ROW.SIGMA ** 2
        self._a = (2 * np.pi * self._sigma2) ** -1.5
        self._ROW = ROW

    def __getattr__(self, name):
        return getattr(self._ROW, name)


class GaussianSourceFEM(GaussianSourceBase):
    _BRAIN_R = 7.9
    NECK_ANGLE = -np.pi / 3
    NECK_AT = _BRAIN_R * np.sin(NECK_ANGLE)

    def csd(self, X, Y, Z):
        DIST2 = (X*X + Y*Y + Z*Z)
        return np.where((DIST2 <= self._BRAIN_R ** 2) & (Y > self.NECK_AT),
                        self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2),
                        0)

    def potential(self, electrodes):
        return self._ROW.loc[electrodes]


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

    def init(self, x, y, z, ROW):
        super(GaussianSourceKCSD3D, self).init(x, y, z, ROW)
        self.conductivity = ROW.CONDUCTIVITY
        self._b = 0.25 / (np.pi * ROW.CONDUCTIVITY)
        self._c = np.sqrt(0.5) / ROW.SIGMA

    def csd(self, X, Y, Z):
        return self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2)

    def potential(self, electrodes):
        R = np.sqrt((electrodes.X - self.x) ** 2 + (electrodes.Y - self.y) ** 2 + (electrodes.Z - self.z) ** 2)
        Rc = R * self._c
        return self._b * np.where(Rc >= self._radius_of_erf_to_x_limit_applicability,
                                  erf(Rc) / R,
                                  self._c * self._fraction_of_erf_to_x_limit_in_0)


class PolarGaussianSourceFEM(PolarBase, GaussianSourceFEM):
    pass


class PolarGaussianSourceKCSD3D(PolarBase, GaussianSourceKCSD3D):
    pass


class ElectrodeAwarePolarGaussianSourceKCSD3D(ElectrodeAware,
                                              PolarGaussianSourceKCSD3D):
    pass


class CartesianGaussianSourceKCSD3D(CartesianBase, GaussianSourceKCSD3D):
    pass


class ElectrodeAwareCartesianGaussianSourceKCSD3D(ElectrodeAware,
                                                  CartesianGaussianSourceKCSD3D):
    pass


def cv(reconstructor, measured, regularization_parameters):
    errors = []

    for regularization_parameter in regularization_parameters:
        logger.info('cv(): error estimation for regularization parameter: {:g}'.format(regularization_parameter))
        ERR = np.array(reconstructor.leave_one_out_errors(measured,
                                                          regularization_parameter))
        errors.append(np.sqrt((ERR**2).mean()))

    return errors

try:
    import pandas as pd

except ImportError:
    pass

else:
    LoadedPotentials = collections.namedtuple('LoadedPotentials',
                                              ['POTENTIALS',
                                               'ELECTRODES'])

    def loadPotentialsAdHocNPZ(filename, conductivity=None):
        fh = np.load(filename)
        ELECTRODES = fh['ELECTRODES']
        ELECTRODE_NAMES = ['E{{:0{}d}}'.format(max(int(np.ceil(np.log10(ELECTRODES.shape[1]))),
                                                   3)).format(i + 1)
                           for i in range(ELECTRODES.shape[1])]
        ELECTRODES = pd.DataFrame(ELECTRODES.T, columns=['X', 'Y', 'Z'],
                                  index=ELECTRODE_NAMES)

        POTENTIALS = pd.DataFrame(fh['POTENTIAL'], columns=ELECTRODES.index)
        for k in ['RES', 'DEGREE', 'SIGMA', 'X', 'Y', 'Z', 'INTEGRAL',
                  'R', 'AZIMUTH', 'ALTITUDE', 'CONDUCTIVITY', 'TIME', 'N',
                  'WRAP', 'MAX_ITER', 'ITER']:
            if k in fh:
                POTENTIALS[k] = fh[k]

        if 'ITER' not in POTENTIALS and 'MAX_ITER' in POTENTIALS:
            POTENTIALS['ITER'] = POTENTIALS.pop('MAX_ITER')

        if conductivity is not None:
            POTENTIALS['CONDUCTIVITY'] = conductivity

        elif 'CONDUCTIVITY' in fh:
            POTENTIALS['CONDUCTIVITY'] = fh['CONDUCTIVITY']

        return LoadedPotentials(POTENTIALS, ELECTRODES)



# MOI sources


class _MethodOfImagesSourceBase(object):
    SourceConfig = collections.namedtuple('SourceConfig',
                                          ['X',
                                           'Y',
                                           'Z',
                                           'SIGMA',
                                           'CONDUCTIVITY'])

    def __init__(self, mask_invalid_space):
        self.mask_invalid_space = mask_invalid_space

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

    def potential(self, electrodes):
        return np.where(self.is_applicable(electrodes.X,
                                           electrodes.Y,
                                           electrodes.Z),
                        self._calculate_field('potential', electrodes),
                        np.nan)


class InfiniteSliceSource(_MethodOfImagesSourceBase):
    """
    Torbjorn V. Ness (2015)
    """

    def __init__(self, y, sigma, h, brain_conductivity, saline_conductivity,
                 glass_conductivity=0,
                 n=20,
                 x=0,
                 z=0,
                 SourceClass=CartesianGaussianSourceKCSD3D,
                 mask_invalid_space=True):
        super(InfiniteSliceSource, self).__init__(mask_invalid_space)
        self.x = x
        self.y = y
        self.z = z
        self.h = h

        wtg = float(brain_conductivity - glass_conductivity) / (brain_conductivity + glass_conductivity)
        wts = float(brain_conductivity - saline_conductivity) / (brain_conductivity + saline_conductivity)
        self.n = n
        self._source = SourceClass(
                           self.SourceConfig(x, y, z, sigma, brain_conductivity))
        weights = [1]
        sources = [self._source]
        for i in range(n):
            weights.append(wtg**i * wts**(i+1))
            sources.append(SourceClass(
                               self.SourceConfig(x,
                                                 2 * (i+1) * h - y,
                                                 z,
                                                 sigma,
                                                 brain_conductivity)))
            weights.append(wtg**(i+1) * wts**i)
            sources.append(SourceClass(
                               self.SourceConfig(x,
                                                 -2 * i * h - y,
                                                 z,
                                                 sigma,
                                                 brain_conductivity)))

        for i in range(1, n + 1):
            weights.append((wtg * wts)**i)
            weights.append((wtg * wts)**i)
            sources.append(SourceClass(
                               self.SourceConfig(x,
                                                 y + 2 * i * h,
                                                 z,
                                                 sigma,
                                                 brain_conductivity)))

            sources.append(SourceClass(
                               self.SourceConfig(x,
                                                 y - 2 * i * h,
                                                 z,
                                                 sigma,
                                                 brain_conductivity)))

        self._positive = [(w, s) for w, s in zip(weights, sources)
                          if w > 0]
        self._positive.sort(key=operator.itemgetter(0), reverse=False)
        self._negative = [(w, s) for w, s in zip(weights, sources)
                          if w < 0]
        self._negative.sort(key=operator.itemgetter(0), reverse=True)

    def is_applicable(self, X, Y, Z):
        return (Y >= 0) & (Y < self.h)

    def _calculate_field(self, name, *args, **kwargs):
        return (sum(w * getattr(s, name)(*args, **kwargs) for w, s in self._positive)
                + sum(w * getattr(s, name)(*args, **kwargs) for w, s in self._negative))


class ElectrodeAwareInfiniteSliceSource(ElectrodeAware, InfiniteSliceSource):
    pass


class HalfSpaceSource(_MethodOfImagesSourceBase):
    def __init__(self, y, sigma, brain_conductivity, saline_conductivity,
                 x=0,
                 z=0,
                 SourceClass=CartesianGaussianSourceKCSD3D,
                 mask_invalid_space=True):
        super(HalfSpaceSource, self).__init__(mask_invalid_space)
        self.x = x
        self.y = y
        self.z = z

        self._source = SourceClass(self.SourceConfig(x, y, z, sigma,
                                                     brain_conductivity))

        self._image = SourceClass(self.SourceConfig(x, -y, z, sigma,
                                                    brain_conductivity))
        self._weight = float(brain_conductivity - saline_conductivity) / (
                brain_conductivity + saline_conductivity)

    def is_applicable(self, X, Y, Z):
        return Y < 0

    def _calculate_field(self, name, *args, **kwargs):
        return (getattr(self._source, name)(*args, **kwargs)
                + self._weight * getattr(self._image, name)(*args, **kwargs))


class ElectrodeAwareHalfSpaceSource(ElectrodeAware, HalfSpaceSource):
    pass



# FEM sources


class GaussianFEM(object):
    def __init__(self, ELECTRODES, FEM, FEM_ELECTRODES,
                 x, y, z, sigma,
                 _y_min=-np.inf,
                 _y_max=np.inf):
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma
        self.y_min = _y_min
        self.y_max = _y_max

        assert isinstance(sigma, float)

        # # DIRTY HACK:
        # # I use np.abs(FEM.Y - y) < sigma / 2 as for some reason there is no source at
        # TMP = FEM[(FEM.SIGMA == sigma) & (np.abs(FEM.Y - y) < sigma / 2)].groupby('R', sort=True).mean()
        # TMP = FEM[(FEM.SIGMA == sigma) & (FEM.Y == y)].sort_values('R')
        TMP = FEM[(FEM.SIGMA == sigma) & (np.abs(FEM.Y - y) < sigma * 1e-10)].sort_values('R')
        interpolate = {el_y: interp1d(TMP.R.values,
                                                 TMP[name].values,
                                                 'linear')
                       for name, el_y in zip(FEM_ELECTRODES.index,
                                             FEM_ELECTRODES.Y)
                       }
        self.ELECTRODES = pd.DataFrame({'R': np.sqrt((self.x - ELECTRODES.X) ** 2
                                                   + (self.z - ELECTRODES.Z) ** 2),
                                        'INTERPOLATE': ELECTRODES.Y.apply(interpolate.__getitem__),
                                        })

    def potential(self, electrodes):
        return self.ELECTRODES.loc[electrodes].apply(lambda ROW: ROW.INTERPOLATE(ROW.R),
                                                     axis=1)

    def csd(self, X, Y, Z):
        return np.where((Y < self.y_max) & (Y > self.y_min),
                        np.exp(-0.5 / self.sigma**2 * ((X - self.x) ** 2
                                                       + (Y - self.y) ** 2
                                                       + (Z - self.z) ** 2)),
                        0) * (2 * np.pi * self.sigma ** 2) **-1.5
