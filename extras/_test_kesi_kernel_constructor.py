#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

import itertools

import numpy as np
import scipy.integrate as si

from kesi import common

from kesi.kernel import constructor, pbf


if __name__ == '__main__':
    ns = [6 * i for i in range(1, 4)]
    pot_grid = [np.linspace(-1, 1, n // 3 + 1) for n in ns]
    csd_grid = [np.linspace(-1, 1, n // 2 + 1) for n in ns]
    src_grid = [np.linspace(-1, 1, n // 6 + 1) for n in ns]

    conv = constructor.Convolver(pot_grid, csd_grid)

    for name, expected_grid in [('POT', pot_grid),
                                ('CSD', csd_grid),
                                ('SRC', src_grid)]:
        observed_grid = getattr(conv, f'{name}_GRID')

        for i, (coord, expected) in enumerate(zip(['X', 'Y', 'Z'],
                                                  expected_grid)):
            observed = getattr(conv, f'{name}_{coord}')

            assert (observed == observed_grid[i]).all()
            assert observed.shape == observed_grid[i].shape

            assert (observed.flatten() == expected).all()
            for j, n in enumerate(observed.shape):
                assert n == (1 if j != i else len(expected))


    def csd(x, y, z):
        return np.maximum(1 - (x ** 2 + y ** 2 + z ** 2) * 4, 0)


    for i_x in range(conv.SRC_X.shape[0]):
        for i_y in range(conv.SRC_Y.shape[1]):
            for i_z in range(conv.SRC_Z.shape[2]):
                SRC = np.zeros(
                    [S.shape[i] for i, S in enumerate(conv.SRC_GRID)])
                SRC[i_x, i_y, i_z] = 2
                CSD = conv.basis_functions_weights_to_csd(SRC, csd, [3, 3, 5])
                EXP = 2 * csd(conv.CSD_X - conv.SRC_X[i_x, 0, 0],
                              conv.CSD_Y - conv.SRC_Y[0, i_y, 0],
                              conv.CSD_Z - conv.SRC_Z[0, 0, i_z])
                ERR = CSD - EXP
                assert abs(ERR).max() < 1e-11

    LEADFIELD = np.ones([S.shape[i] for i, S in enumerate(conv.POT_GRID)])
    LEADFIELD[conv.src_idx('POT')] -= 1
    acc = 8
    for d in conv.steps('POT'):
        acc *= d

    POT = conv.leadfield_to_potential_basis_functions(LEADFIELD,
                                                      lambda x, y, z: 1 / acc,
                                                      [[0.5, 0, 0.5]] * 3)

    for (wx, idx_x), (wy, idx_y), (wz, idx_z) in itertools.product([(0.5, 0),
                                                                    (1, slice(1,
                                                                              -1)),
                                                                    (0.5, -1)],
                                                                   repeat=3):
        assert (abs(POT[idx_x, idx_y, idx_z] - wx * wy * wz) < 1e-11).all()

    grid = [np.linspace(-1.1, -1, 100)] * 3
    conv = constructor.Convolver(grid, grid)

    conductivity = 0.33
    LEADFIELD = 0.25 / np.pi / conductivity / np.sqrt(np.square(conv.POT_X)
                                                      + np.square(conv.POT_Y)
                                                      + np.square(conv.POT_Z))
    WEIGHTS = si.romb(np.identity(65)) / 64

    sd = conv.steps('POT')[0] * 64 / 6
    model_src = common.SphericalSplineSourceKCSD(0, 0, 0,
                                                 [sd, 3 * sd],
                                                 [[1],
                                                  [0,
                                                   2.25 / sd,
                                                   -1.5 / sd ** 2,
                                                   0.25 / sd ** 3]],
                                                 conductivity)

    POT_GT = model_src.potential(*conv.SRC_GRID)
    POT = conv.leadfield_to_potential_basis_functions(LEADFIELD, model_src.csd,
                                                      [WEIGHTS] * 3)
    IDX = (slice(32, 100 - 32),) * 3

    assert abs(POT / POT_GT - 1)[IDX].max() < 1e-5


    ECHO = True
    CONDUCTIVITY = 0.3
    R = 1.0
    ROMBERG_K = 6
    ROMBERG_N = 2 ** ROMBERG_K + 1

    class TestElectrodeKESI(object):
        x = R
        y = R * 0.5
        z = R * 2
        base_conductivity = CONDUCTIVITY
        dx = 2 * R / (ROMBERG_N - 1)

        def correction_leadfield(self, X, Y, Z):
            return (0.25 / np.pi / self.base_conductivity
                   * np.power(np.square(X + self.x)
                              + np.square(Y - self.y)
                              + np.square(Z - self.z),
                              -0.5))

        def base_leadfield(self, X, Y, Z):
            return (0.25 / np.pi / self.base_conductivity
                    / (0.15 * self.dx
                       + np.sqrt(np.square(X - self.x)
                                 + np.square(Y - self.y)
                                 + np.square(Z - self.z))))

        def leadfield(self, X, Y, Z):
            return (self.base_leadfield(X, Y, Z)
                    + self.correction_leadfield(X, Y, Z))


    class TestElectrodeKCSD(object):
        x = R
        y = R * 0.5
        z = R * 2
        base_conductivity = CONDUCTIVITY
        dx = 2 * R / (ROMBERG_N - 1)

        def leadfield(self, X, Y, Z):
            return (0.25 / np.pi / self.base_conductivity
                    / (0.15 * self.dx
                       + np.sqrt(np.square(X - self.x)
                                 + np.square(Y - self.y)
                                 + np.square(Z - self.z))))


    def get_source(x=0, y=0, z=0):
        return common.GaussianSourceKCSD3D(x, y, z, R / 4, CONDUCTIVITY)


    def assertRelativeErrorWithinTolerance(expected, observed,
                                           tolerance=0,
                                           echo=False):
        max_error = abs(observed / expected - 1).max()
        if echo:
            print(max_error)
        assert max_error <= tolerance


    test_electrode_kcsd = TestElectrodeKCSD()
    test_electrode_kesi = TestElectrodeKESI()
    model_src = get_source()
    X = np.linspace(R, 9 * R, 2 ** (ROMBERG_K + 2) + 1)
    Y = np.linspace(-1.5 * R, 2.5 * R, 2 * 2 ** (ROMBERG_K + 1) + 1)
    Z = np.linspace(0, 4 * R, 4 * 2 ** (ROMBERG_K + 1) + 1)

    convolver = constructor.Convolver([X, Y, Z], [X, Y, Z])
    romberg_weights = tuple(si.romb(np.identity(2 ** _k + 1)) / 2 ** _k
                            for _k in range(ROMBERG_K, ROMBERG_K + 3))

    SRC_MASK = ((convolver.SRC_X >= 2 * R) & (convolver.SRC_X <= 8 * R)) & (
                (convolver.SRC_Y >= -0.5 * R) & (
                    convolver.SRC_Y <= 1.5 * R)) & (
                     (convolver.SRC_Z >= R) & (convolver.SRC_Z <= 3 * R))

    convolver_interface = constructor.ConvolverInterfaceIndexed(convolver,
                                                                model_src.csd,
                                                                romberg_weights,
                                                                SRC_MASK)


    MASK_XY = (np.ones_like(convolver.SRC_X, dtype=bool)
               & np.ones_like(convolver.SRC_Y, dtype=bool))
    MASK_MINOR = (MASK_XY & (convolver.SRC_Z > 2 * R))
    MASK_MAJOR = ~MASK_MINOR

    # kCSD
    reciprocal_src = get_source(test_electrode_kesi.x,
                                test_electrode_kesi.y,
                                test_electrode_kesi.z)
    expected = reciprocal_src.potential(convolver.SRC_X,
                                        convolver.SRC_Y,
                                        convolver.SRC_Z)[SRC_MASK]
    # kCSD analytical

    tested = pbf.Analytical(convolver_interface,
                            potential=model_src.potential)
    with tested:
        observed = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-10, ECHO)

    # kCSD numeric
    tested = pbf.Numerical(convolver_interface)
    with tested:
        observed = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-2, ECHO)

    # kCSD masked
    # kCSD masked analytical
    tested = pbf.AnalyticalMaskedNumerically(convolver_interface,
                                             potential=model_src.potential,
                                             leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kcsd)
    tested = pbf.AnalyticalMaskedNumerically(convolver_interface,
                                             potential=model_src.potential,
                                             leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)

    # kCSD masked numerical
    tested = pbf.NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kcsd)
    tested = pbf.NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)

    # kESI
    correction = reciprocal_src.potential(-convolver.SRC_X,
                                          convolver.SRC_Y,
                                          convolver.SRC_Z)[SRC_MASK]

    # kESI correction
    tested = pbf.NumericalCorrection(convolver_interface)
    with tested:
        observed = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(correction, observed, 2e-4, ECHO)


    expected += correction

    # kESI analytical
    tested = pbf.AnalyticalCorrectedNumerically(convolver_interface,
                                                potential=model_src.potential)
    with tested:
        observed = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-4, ECHO)

    # kESI numerical
    tested = pbf.Numerical(convolver_interface)
    with tested:
        observed = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-2, ECHO)

    # kESI analytical masked
    tested = pbf.AnalyticalMaskedAndCorrectedNumerically(
                                              convolver_interface,
                                              potential=model_src.potential,
                                              leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kesi)
    tested = pbf.AnalyticalMaskedAndCorrectedNumerically(
                                              convolver_interface,
                                              potential=model_src.potential,
                                              leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)

    # kESI numerical masked
    tested = pbf.NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kesi)
    tested = pbf.NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)
