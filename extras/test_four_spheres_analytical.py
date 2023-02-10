#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
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

import sys

import numpy as np
import pandas as pd

from dolfin import assemble, Constant, Expression, grad, inner

from _common_new import FourSphereModel
import FEM.fem_common as fc

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt
import cbf


CONFIG = 'FEM/model_properties/four_spheres_csf_3_mm.ini'
MESH = sys.argv[1]
DEGREE = int(sys.argv[2])


CONDUCTIVITY = FourSphereModel.Properties.from_config(CONFIG, 'conductivity')
RADIUS = FourSphereModel.Properties.from_config(CONFIG, 'radius')

SCALP_R = RADIUS.scalp

DIPOLE_R = 78e-3
DIPOLE_LOC = np.array([[0., 0., DIPOLE_R]])
DIPOLE_P = np.array([[1e-2, 0., 0.]])

N = 1000

_XY_R = np.sqrt(SCALP_R ** 2 - DIPOLE_R ** 2)
ELECTRODES = pd.DataFrame({'X': np.linspace(-_XY_R, _XY_R, N)})
ELECTRODES['Y'] = 0.0
ELECTRODES['Z'] = DIPOLE_R

ELECTRODES_LOC = np.transpose([ELECTRODES[c] for c in 'XYZ'])

#np.random.seed(42)
#ELECTRODES = pd.DataFrame({
#    'PHI': np.random.uniform(-np.pi, np.pi, N),
#    'THETA': 2 * np.arcsin(np.sqrt(np.random.uniform(0, 1, N))) - np.pi / 2,
#    'R': np.random.uniform(DIPOLE_R, SCALP_R), 
#    })
#
#ELECTRODES['X'] = ELECTRODES.R * np.sin(ELECTRODES.THETA)
#_XY_R = ELECTRODES.R * np.cos(ELECTRODES.THETA)
#ELECTRODES['Y'] = _XY_R * np.cos(ELECTRODES.PHI)
#ELECTRODES['Z'] = _XY_R * np.sin(ELECTRODES.PHI)


class SubtractionDipoleSourcePotentialFEM(fc._SubtractionPointSourcePotentialFEM):
    MAX_ITER = 1000

    def correction_potential(self, loc, p):
        with self.local_preprocessing_time:
            logger.debug('Creating RHS...')
            L = self._rhs(loc, p)
            logger.debug('Done.  Assembling linear equation vector...')
            self._known_terms = assemble(L)
            logger.debug('Done.  Assembling linear equation matrix...')
            self._terms_with_unknown = assemble(self._a)
            logger.debug('Done.')
            self._modify_linear_equation(loc, p)

        try:
            logger.debug('Solving linear equation...')
            with self.solving_time:
                self._solve()

            logger.debug('Done.')
            return self._potential_function

        except RuntimeError as e:
            self.iterations = self.MAX_ITER
            logger.warning("Solver failed: {}".format(repr(e)))
            return None

    def _modify_linear_equation(self, loc, p):
        pass

    def _rhs(self, loc, p):
        x, y, z = loc
        base_conductivity = self.base_conductivity(loc, p)
        self._setup_expression(self._base_potential_expression,
                               base_conductivity, loc, p)
        self._setup_expression(self._base_potential_gradient_normal_expression,
                               base_conductivity, loc, p)
        # Eq. 20 at Piastra et al 2018
        return (-sum((inner((Constant(c - base_conductivity)
                             * grad(self._base_potential_expression)),
                            grad(self._v))
                      * self._dx(x)
                      for x, c in self.CONDUCTIVITY
                      if c != base_conductivity))
                - sum(Constant(base_conductivity)
                      # * inner(self._facet_normal,
                      #         grad(self._base_potential_expression))
                      * self._base_potential_gradient_normal_expression
                      * self._v
                      * self._ds(s)
                      for s, _ in self.BOUNDARY_CONDUCTIVITY))

    def _setup_expression(self, expression, base_conductivity, loc, p):
        expression.conductivity = base_conductivity
        expression.loc_x, expression.loc_y, expression.loc_z = loc
        expression.p_x, expression.p_y, expression.p_z = p

    def _potential_gradient_normal(self, conductivity=0.0):
        rx = '(x[0] - loc_x)'
        ry = '(x[1] - loc_y)'
        rz = '(x[2] - loc_z)'
        r2 = f'({rx} * {rx} + {ry} * {ry} + {rz} * {rz})'
        r5 = f'pow({r2}, 2.5)'
        p_r = f'({rx} * p_x + {ry} * p_y + {rz} * p_z)'
        r_sphere = 'sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])'
        dot_p = f'(x[0] * p_x + x[1] * p_y + x[2] * p_z)'
        dot_r = f'(x[0] * {rx} + x[1] * {ry} + x[2] * {rz})'
        return Expression(f'''
                           {0.25 / np.pi}
                           * ({r2} * {dot_p}
                              - 3 * {p_r} * {dot_r})
                           / ({r5} * {r_sphere} * conductivity)
                           ''',
                           degree=self.degree,
                           domain=self._fm.mesh,
                           loc_x=0.0,
                           loc_y=0.0,
                           loc_z=0.0,
                           p_x=0.0,
                           p_y=0.0,
                           p_z=0.0,
                           conductivity=conductivity)

    def _potential_expression(self, conductivity=0.0):
        rx = '(x[0] - loc_x)'
        ry = '(x[1] - loc_y)'
        rz = '(x[2] - loc_z)'
        r3 = f'pow({rx} * {rx} + {ry} * {ry} + {rz} * {rz}, 1.5)'
        p_r = f'({rx} * p_x + {ry} * p_y + {rz} * p_z)'
        return Expression(f'''
                           {0.25 / np.pi} 
                           * {p_r}
                           / ({r3} * conductivity)
                           ''',
                           degree=self.degree,
                           domain=self._fm.mesh,
                           loc_x=0.0,
                           loc_y=0.0,
                           loc_z=0.0,
                           p_x=0.0,
                           p_y=0.0,
                           p_z=0.0,
                           conductivity=conductivity)

    def base_conductivity(self, loc, p):
        return self.config.getfloat('brain', 'conductivity')


analyticalModel = FourSphereModel(CONDUCTIVITY, RADIUS)
analyticalDipole = analyticalModel(DIPOLE_LOC, DIPOLE_P)

fem = SubtractionDipoleSourcePotentialFEM(fc.FunctionManager(MESH, DEGREE, 'CG'),
                                          CONFIG)

def potential_base(loc, p, conductivity, X):
    R = np.reshape(X, (-1, 3)) - np.reshape(loc, (1, 3))
    RADIUS = np.sqrt(np.square(R).sum(axis=1)).reshape(-1, 1)
    return (0.25 / (np.pi * conductivity)
            / RADIUS ** 3
            * np.matmul(R, np.reshape(p, (3, 1))))


potential_correction = fem.correction_potential(DIPOLE_LOC.flatten(),
                                                DIPOLE_P.flatten())



ELECTRODES['BASE_POTENTIAL'] = potential_base(DIPOLE_LOC,
                                              DIPOLE_P,
                                              analyticalModel.conductivity.brain,
                                              ELECTRODES_LOC).flatten()

_TMP = []
for x, y, z in ELECTRODES_LOC:
    try:
        _TMP.append(potential_correction(x, y, z))
    except RuntimeError:
        _TMP.append(np.nan)

ELECTRODES['FEM_CORRECTION'] = _TMP
#ELECTRODES['FEM_CORRECTION'] = [potential_correction(x, y, z)
#                                for x, y, z in zip(*[ELECTRODES[c] / 100 for c in 'XYZ'])]

ELECTRODES['FEM_POTENTIAL'] = ELECTRODES.BASE_POTENTIAL + ELECTRODES.FEM_CORRECTION
ELECTRODES['ANALYTICAL_POTENTIAL'] = analyticalDipole(*[ELECTRODES[c] for c in 'XYZ']).flatten()


#plt.ion()
fig, axes = plt.subplots(2, 1,
                         gridspec_kw=dict(height_ratios=[2, 1]))
for ax, scale in zip(axes, ['symlog', 'linear']):
    if scale == 'symlog':
        ax.set_yscale(scale, linthresh=1)
    else:
        ax.set_yscale(scale)

        if scale == 'linear':
            ax.set_ylim(-10, 10)

    for y in [-1, 1]:
        ax.axhline(y, ls=':', color=cbf.BLACK)

    for r in RADIUS:
        x = np.sqrt(r ** 2 - DIPOLE_R ** 2)
        ax.axvline(x, ls=':', color=cbf.BLACK)
        ax.axvline(-x, ls=':', color=cbf.BLACK)

    ax.plot(ELECTRODES.X, ELECTRODES.BASE_POTENTIAL,
            color=cbf.YELLOW,
            ls='-',
            label='kCSD approximation')
    ax.plot(ELECTRODES.X, ELECTRODES.FEM_POTENTIAL,
            ls='-',
            color=cbf.SKY_BLUE,
            label='FEM-corrected')
    ax.plot(ELECTRODES.X, ELECTRODES.ANALYTICAL_POTENTIAL,
            ls='--',
            color=cbf.VERMILION,
            label='Analitical')

    ax.legend(loc='best')

plt.show()
