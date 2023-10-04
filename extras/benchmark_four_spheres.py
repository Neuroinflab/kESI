#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2023 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#    Copyright (C) 2023 Jakub M. Dzik (Institute of Applied Psychology;       #
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

import collections
import warnings
import configparser

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
from scipy.special import lpmv

import pandas as pd
import matplotlib.pyplot as plt

from local.fem._common import Stopwatch

from kesi.common import FourSphereModel as ModelNew


CONFIG = 'FEM/model_properties/four_spheres_csf_3_mm.ini'
DIPOLE_R = 78e-3
DIPOLE_LOC = [0., 0., DIPOLE_R]
DIPOLE_P = [0.002, 0.003, 0.005]

EXPECTED_TIME_LIMIT = 4 * 3600  # [s]


class ModelOld(object):
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
                         for section in ModelOld._LAYERS])

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
                print("Invalid electrode position: {:f} (off by {:e})".format(r_ele, r_ele - self.radius.scalp))
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



K = np.arange(6, 15)
N = 2 ** K
N[-1] = 14_900  # 14_901+ yields errors

def getDipoles(n):
    return tuple(c.from_config(CONFIG, n)(DIPOLE_LOC, DIPOLE_P)
                 for c in [ModelOld, ModelNew])

dipoles = [getDipoles(n) for n in N]

SCALP_R = dipoles[0][0].model.radius.scalp
xy_r = np.sqrt(SCALP_R ** 2 - DIPOLE_R ** 2)

oldSW = Stopwatch()
newSW = Stopwatch()

DF = []
for k_ele in range(21):
    n_ele = 2 ** k_ele

    ELECTRODES = pd.DataFrame({'R': np.linspace(-xy_r, xy_r, n_ele)})
    ELECTRODES['X'] = np.sin(np.pi / 3) * ELECTRODES.R
    ELECTRODES['Y'] = np.cos(np.pi / 3) * ELECTRODES.R
    ELECTRODES['Z'] = DIPOLE_R

    ELECTRODES_LOC = np.transpose([ELECTRODES[c] for c in 'XYZ'])

    for k, n, (oldDipole, newDipole) in zip(K, N, dipoles):
        logger.info(f"{n_ele}\t{n}")

        row =   {'K': k,
                 'N': n,
                 'K_ELE': k_ele,
                 'N_ELE': n_ele,
                 }
        DF.append(row)

        oldRes = newRes = None

        if np.log2(n) * 1.5 + np.log2(n_ele) - 20.5 > np.log2(EXPECTED_TIME_LIMIT):
            logger.info("  skipping old")  # protection against long calculations (fitted)

        else:
            with oldSW:
                oldRes = oldDipole(*ELECTRODES_LOC.T)

            logger.info(f"{n_ele}\t{n}\tOLD: {float(oldSW):.1e}\tNaNs: {np.isnan(oldRes).sum()}\tInfs: {np.isinf(oldRes).sum()}")
            row.update(T_OLD=float(oldSW),
                       NANS_OLD=np.isnan(oldRes).sum(),
                       INFS_OLD=np.isinf(oldRes).sum())

        if np.log2(n) * 2 + np.log2(n_ele) - 27 > np.log2(EXPECTED_TIME_LIMIT):
            logger.info("  skipping new")  # protection against long calculations (fitted)

        else:
            with newSW:
                newRes = newDipole(*ELECTRODES_LOC.T)

            logger.info(f"{n_ele}\t{n}\tNEW: {float(newSW):.1e}\tNaNs: {np.isnan(newRes).sum()}\tInfs: {np.isinf(newRes).sum()}")
            row.update(T_NEW=float(newSW),
                       NANS_NEW=np.isnan(newRes).sum(),
                       INFS_NEW=np.isinf(newRes).sum())

        if oldRes is not None and newRes is not None:
            AVG = 0.5 * (oldRes + newRes)
            DIFF = oldRes - newRes
            DIFF_REL = DIFF / AVG

            row.update(DIFF_L1=abs(DIFF).mean(),
                       DIFF_L2=np.sqrt(np.square(DIFF).mean()),
                       DIFF_Linf=abs(DIFF).max(),
                       DIFF_REL_L1=abs(DIFF_REL).mean(),
                       DIFF_REL_L2=np.sqrt(np.square(DIFF_REL).mean()),
                       DIFF_REL_Linf=abs(DIFF_REL).max(),
                       L1=abs(AVG).mean(),
                       L2=np.sqrt(np.square(AVG).mean()),
                       Linf=abs(AVG).max())


DF = pd.DataFrame(DF)
DF.to_csv('benchmark_four_spheres.csv', index=False)

TMP = DF.loc[DF.K_ELE == 10]

plt.plot(TMP.N, TMP.T_OLD, marker='o',  label='old')
plt.plot(TMP.N, TMP.T_NEW, marker='x',  label='new')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.legend(loc='best')
plt.ylabel('time [s]')
plt.axhline(1, ls=':', color='k')
plt.axhline(60, ls=':', color='k')
plt.axhline(3600, ls=':', color='k')
plt.title('1024 electrodes')


for k in [6, 9, 12, 13, 14]:
    n, = N[K == k]
    TMP = DF.loc[DF.K == k]

    plt.figure()
    plt.plot(TMP.N_ELE, TMP.T_OLD, marker='o',  label='old')
    plt.plot(TMP.N_ELE, TMP.T_NEW, marker='x',  label='new')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n electrodes')
    plt.legend(loc='best')
    plt.ylabel('time [s]')
    plt.axhline(1, ls=':', color='k')
    plt.axhline(60, ls=':', color='k')
    plt.axhline(3600, ls=':', color='k')
    plt.title(f'n = {n}')
