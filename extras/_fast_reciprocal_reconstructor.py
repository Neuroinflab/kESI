#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
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

import functools
import warnings

import numpy as np
import scipy.integrate as si
import scipy.signal as ssi


def deprecated(old, new):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            warnings.warn(
                DeprecationWarning(f"{old} is deprecated, use {new} instead"),
                stacklevel=2)
            return f(*args, **kwargs)

        return wrapper

    return decorator


def shape(axis, n=3):
    return [-1 if i == axis else 1
            for i in range(n)]


def reshape(A, axis, n=3):
    return np.reshape(A, shape(axis, n))


class Convolver(object):
    def __init__(self, potential_grid, csd_grid):
        self.POT_GRID = []
        self.CSD_GRID = []
        self.SRC_GRID = []

        for i, (c, POT, CSD) in enumerate(zip(['X', 'Y', 'Z'],
                                              potential_grid,
                                              csd_grid)):
            POT = reshape(POT, i)
            CSD = reshape(CSD, i)

            SRC, CSD_IDX, POT_IDX = np.intersect1d(
                CSD.flatten(),
                POT.flatten(),
                assume_unique=True,
                return_indices=True)
            SRC = reshape(SRC, i)

            self.POT_GRID.append(POT)
            self.CSD_GRID.append(CSD)
            self.SRC_GRID.append(SRC)

            setattr(self, f'CSD_{c}', CSD)
            setattr(self, f'POT_{c}', POT)
            setattr(self, f'SRC_{c}', SRC)
            setattr(self, f'_SRC_CSD_IDX_{c}', CSD_IDX)
            setattr(self, f'_SRC_POT_IDX_{c}', POT_IDX)

    def leadfield_to_base_potentials(self,
                                     LEADFIELD,
                                     csd,
                                     weights):

        steps = self.steps('POT')

        ns = list(map(len, weights))

        CSD = self.csd_kernel(csd, ns, steps)

        WEIGHTS = functools.reduce(np.multiply,
                                   [d * (n - 1) * reshape(w, i)
                                    for i, (w, n, d) in enumerate(zip(weights,
                                                                      ns,
                                                                      steps))])
        return ssi.fftconvolve(LEADFIELD,
                               CSD * WEIGHTS,
                               mode='same')[self.src_idx('POT')]

    def base_weights_to_csd(self, BASE_WEIGHTS, csd, csd_ns):
        if np.shape(BASE_WEIGHTS) != self.csd_shape:
            BW = np.zeros(self.csd_shape)
            BW[self.src_idx('CSD')] = BASE_WEIGHTS
        else:
            BW = BASE_WEIGHTS

        return ssi.fftconvolve(BW,
                               self.csd_kernel(csd, csd_ns, self.steps('CSD')),
                               mode='same')

    def csd_kernel(self, csd, ns, steps):
        return csd(*np.meshgrid(*[np.array([0.]) if np.isnan(h) else
                                  h * np.linspace(-(n // 2), n // 2, n)
                                  for h, n in zip(steps, ns)],
                                indexing='ij',
                                sparse=True))

    def steps(self, name):
        return [(DIM[-1, -1, -1] - DIM[0, 0, 0]) / (DIM.size - 1)
                for i, DIM in enumerate([getattr(self, f'{name}_{c}')
                                         for c in ['X', 'Y', 'Z']])]

    def src_idx(self, name):
        return np.ix_(*[getattr(self, f'_SRC_{name}_IDX_{c}')
                        for c in ['X', 'Y', 'Z']])

    @property
    def csd_shape(self):
        return self.shape('CSD')

    def shape(self, name):
        return tuple(S.shape[i]
                     for i, S in enumerate(getattr(self, f'{name}_GRID')))


class ckESI_convolver(Convolver):
    @deprecated('class ckESI_convolver', 'Convolver class')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecated('.ds() method', '.steps()')
    def ds(self, name):
        return self.steps()

    @property
    @deprecated('.POT_MESH attribute', '.POT_GRID')
    def POT_MESH(self):
        return self.POT_GRID

    @property
    @deprecated('.CSD_MESH attribute', '.CSD_GRID')
    def CSD_MESH(self):
        return self.CSD_GRID

    @property
    @deprecated('.SRC_MESH attribute', '.SRC_GRID')
    def SRC_MESH(self):
        return self.SRC_GRID


class KernelConstructor(object):
    @staticmethod
    def create_kernel(base_images_at_electrodes):
        return np.matmul(base_images_at_electrodes.T,
                         base_images_at_electrodes)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._base_images

    def create_base_images_at_electrodes(self,
                                         electrodes,
                                         potential_at_electrode):
        with self:
            with potential_at_electrode:
                self._create_base_images_at_electrodes(electrodes,
                                                       potential_at_electrode)

            return self._base_images

    def _create_base_images_at_electrodes(self, electrodes, potential_at_electrode):
        for i, electrode in enumerate(electrodes):
            POT = potential_at_electrode(electrode)

            self._alloc_base_images_if_necessary(POT.size, len(electrodes))
            self._base_images[:, i] = POT

    def _alloc_base_images_if_necessary(self, n_bases, n_electrodes):
        if not hasattr(self, '_base_images'):
            self._base_images = np.full((n_bases, n_electrodes),
                                        np.nan)


class ckESI_kernel_constructor(KernelConstructor):
    @deprecated('class ckESI_kernel_constructor', 'KernelConstructor class')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CrossKernelConstructor(object):
    def __init__(self,
                 convolver_interface,
                 csd_mask,
                 csd_allowed_mask=None):
        self.ci = convolver_interface
        self.csd_mask = csd_mask
        self.csd_allowed_mask = csd_allowed_mask

    def __enter__(self):
        self._base_weights = self.ci.zeros('SRC')

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._base_weights
        for attr in ['_n_electrodes', '_cross_kernel']:
            if hasattr(self, attr):
                delattr(self, attr)

    def __call__(self, base_images_at_electrodes):
        with self:
            self._create_crosskernel(base_images_at_electrodes)
            return self._cross_kernel

    def _create_crosskernel(self, base_images_at_electrodes):
        self._n_electrodes = base_images_at_electrodes.shape[1]
        for i, base_images in enumerate(base_images_at_electrodes.T):
            self.ci.update_src(self._base_weights, base_images)
            self._set_crosskernel_column(i, self._bases_to_csd())

        self._zero_crosskernel_where_csd_not_allowed()

    def _set_crosskernel_column(self, i, column):
        if i == 0:
            self._allocate_cross_kernel(column.size)

        self._cross_kernel[:, i] = column

    def _bases_to_csd(self):
        return self._crop_csd(self.ci.base_weights_to_csd(self._base_weights))

    def _crop_csd(self, csd):
        return csd[self.csd_mask]

    def _allocate_cross_kernel(self, n_points):
        self._cross_kernel = np.full((n_points, self._n_electrodes),
                                     np.nan)

    def _zero_crosskernel_where_csd_not_allowed(self):
        if self.csd_allowed_mask is not None:
            self._cross_kernel[~self._crop_csd(self.csd_allowed_mask), :] = 0


class ckESI_crosskernel_constructor(CrossKernelConstructor):
    @deprecated('class ckESI_crosskernel_constructor',
                'CrossKernelConstructor class')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @deprecated('.csd_indices attribute', '.csd_mask')
    def csd_indices(self):
        return self.csd_mask


class ConvolverInterface_base(object):
    def __init__(self, convolver, csd, weights):
        self.convolver = convolver
        self.csd = csd
        self._set_weights(weights)

    def _set_weights(self, weights):
        self.weights = (weights
                        if isinstance(weights, tuple)
                        else (weights,) * 3)

    @property
    def _src_diameter(self):
        return [len(w) for w in self.weights]

    @property
    def _src_radius(self):
        return [r // 2 for r in self._src_diameter]

    def convolve_csd(self, leadfield):
        return self.convolver.leadfield_to_base_potentials(
            leadfield,
            self.csd,
            list(self.weights))

    def zeros(self, name):
        return np.zeros(self.convolver.shape(name))

    def empty(self, name):
        return np.empty(self.convolver.shape(name))

    def base_weights_to_csd(self, base_weights):
        csd_kernel_shape = [(1 if np.isnan(csd)
                             else int(round(r * pot / csd)) * 2 + 1)
                            for r, pot, csd in zip(self._src_radius,
                                                   *map(self.convolver.steps,
                                                     ['POT', 'CSD']))]
        return self.convolver.base_weights_to_csd(base_weights,
                                                  self.csd,
                                                  csd_kernel_shape)

    def meshgrid(self, name):
        return np.meshgrid(*getattr(self.convolver,
                                    f'{name}_GRID'),
                           indexing='ij')


class ConvolverInterfaceIndexed(ConvolverInterface_base):
    def __init__(self, convolver, csd, weights, source_mask):
        super().__init__(convolver, csd, weights)
        self.source_mask = source_mask

    @property
    @deprecated('.source_indices attribute', '.source_mask')
    def source_indices(self):
        return self.source_mask

    def integrate_source_potential(self, leadfield):
        return self.convolve_csd(leadfield)[self.source_mask]

    def src_coords(self):
        return [A[self.source_mask] for A in self.meshgrid('SRC')]

    def update_src(self, src, values):
        src[self.source_mask] = values


def _sum_of_not_none(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        V, V_SUPER = f(self, *args, **kwargs)
        if V_SUPER is not None:
            V += V_SUPER
        return V
    return wrapper


class _PAE_Base(object):
    def __init__(self, convolver_interface):
        self.convolver_interface = convolver_interface

    def __call__(self, electrode):
        return None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _PAE_PotAttribute(_PAE_Base):
    def __enter__(self):
        super().__enter__()
        self.POT_XYZ = self.convolver_interface.meshgrid('POT')

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.POT_XYZ
        super().__exit__(exc_type, exc_val, exc_tb)


class _PAE_PotProperty(object):
    @property
    def POT_XYZ(self):
        return self.convolver_interface.meshgrid('POT')


class _PAE_FromLeadfield(_PAE_Base):
    @_sum_of_not_none
    def __call__(self, electrode):
        self._create_leadfield(electrode)
        return (self._integrate_source_potential(),
                super().__call__(electrode))

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.LEADFIELD
        super().__exit__(exc_type, exc_val, exc_tb)

    def _integrate_source_potential(self):
        return self.convolver_interface.integrate_source_potential(self.LEADFIELD)


class _PAE_MaskedLeadfield(_PAE_FromLeadfield):
    def __init__(self, convolver_interface, leadfield_allowed_mask, **kwargs):
        super().__init__(convolver_interface, **kwargs)
        self.leadfield_allowed_mask = leadfield_allowed_mask

    def _create_leadfield(self, electrode):
        self._provide_leadfield_array()
        self._modify_leadfield(electrode)

    def _modify_leadfield(self, electrode):
        LEADFIELD = self._allowed_leadfield(electrode)
        if LEADFIELD is not None:
            self.LEADFIELD[self.leadfield_allowed_mask] = LEADFIELD

    def _provide_leadfield_array(self):
        self.clear_leadfield()

    def clear_leadfield(self):
        try:
            self.LEADFIELD.fill(0)

        except AttributeError:
            self.LEADFIELD = self.convolver_interface.zeros('POT')

    def _allowed_leadfield(self, electrode):
        return None


class _PAE_LeadfieldCroppingAnalyticalBasesNumerically(_PAE_MaskedLeadfield):
    """
    `.POT_XYZ` attribute/property required
    """
    def __enter__(self):
        super().__enter__()
        self.csd_forbidden_mask = ~self.leadfield_allowed_mask
        self.POT_XYZ_CROPPED = [A[self.csd_forbidden_mask] for A in self.POT_XYZ]

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.POT_XYZ_CROPPED
        del self.csd_forbidden_mask
        super().__exit__(exc_type, exc_val, exc_tb)

    def _create_leadfield(self, electrode):
        super()._create_leadfield(electrode)
        self.LEADFIELD[self.csd_forbidden_mask] = self._cropping_leadfield(
            electrode)

    def _cropping_leadfield(self, electrode):
        return -getattr(electrode, self._LEADFIELD_METHOD)(*self.POT_XYZ_CROPPED)


class _PAE_PotMaskedAttribute(_PAE_MaskedLeadfield,
                              _PAE_PotProperty):
    """
    `.POT_XYZ` attribute/property required
    """
    def __enter__(self):
        super().__enter__()
        self.POT_XYZ_MASKED = [A[self.leadfield_allowed_mask] for A in self.POT_XYZ]

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.POT_XYZ_MASKED
        super().__exit__(exc_type, exc_val, exc_tb)


class PAE_NumericalMasked(_PAE_PotMaskedAttribute):
    @_sum_of_not_none
    def _allowed_leadfield(self, electrode):
        return (electrode.leadfield(*self.POT_XYZ_MASKED),
                super()._allowed_leadfield(electrode))


class _PAE_FromLeadfieldNotMasked(_PAE_FromLeadfield):
    def _create_leadfield(self, electrode):
        self.LEADFIELD = None


class _PAE_LeadfieldFromElectrode(_PAE_FromLeadfieldNotMasked):
    """
    `.POT_XYZ` attribute required
    """
    def _create_leadfield(self, electrode):
        super()._create_leadfield(electrode)
        LEADFIELD = electrode.leadfield(*self.POT_XYZ)
        if self.LEADFIELD is not None:
            self.LEADFIELD += LEADFIELD
        else:
            self.LEADFIELD = LEADFIELD

# kCSD

class PAE_Analytical(_PAE_Base):
    def __init__(self, convolver_interface, potential, **kwargs):
        super().__init__(convolver_interface, **kwargs)
        self.potential = potential

    def __enter__(self):
        super().__enter__()
        self.SRC_X, self.SRC_Y, self.SRC_Z = self.convolver_interface.src_coords()

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.SRC_X, self.SRC_Y, self.SRC_Z
        super().__exit__(exc_type, exc_val, exc_tb)

    @_sum_of_not_none
    def __call__(self, electrode):
        return (self.potential(electrode.x - self.SRC_X,
                               electrode.y - self.SRC_Y,
                               electrode.z - self.SRC_Z),
                super().__call__(electrode))


class PAE_Numerical(_PAE_LeadfieldFromElectrode,
                    _PAE_PotAttribute):
    pass


class PAE_AnalyticalMaskedNumerically(_PAE_PotProperty,
                                      _PAE_LeadfieldCroppingAnalyticalBasesNumerically,
                                      PAE_Analytical):
    _LEADFIELD_METHOD = 'leadfield'


class _PAE_LeadfieldFromMaskedCorrectionPotential(_PAE_PotMaskedAttribute):
    @_sum_of_not_none
    def _allowed_leadfield(self, electrode):
        return (electrode.correction_leadfield(*self.POT_XYZ_MASKED),
                super()._allowed_leadfield(electrode))


class PAE_AnalyticalMaskedAndCorrectedNumerically(
          _PAE_LeadfieldCroppingAnalyticalBasesNumerically,
          _PAE_LeadfieldFromMaskedCorrectionPotential,
          PAE_Analytical):
    _LEADFIELD_METHOD = 'base_leadfield'

    def _provide_leadfield_array(self):
        self.alloc_leadfield_if_necessary()

    def alloc_leadfield_if_necessary(self):
        if not hasattr(self, 'LEADFIELD'):
            self.LEADFIELD = self.convolver_interface.empty('POT')


class PAE_NumericalCorrection(_PAE_FromLeadfieldNotMasked,
                              _PAE_PotAttribute):
    def _create_leadfield(self, electrode):
        super()._create_leadfield(electrode)
        LEADFIELD = electrode.correction_leadfield(*self.POT_XYZ)
        if self.LEADFIELD is not None:
            self.LEADFIELD += LEADFIELD
        else:
            self.LEADFIELD = LEADFIELD


class PAE_AnalyticalCorrectedNumerically(PAE_NumericalCorrection,
                                         PAE_Analytical):
    pass


# DEPRECATED

class PAE_kCSD_Analytical(PAE_Analytical):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_Analytical class is \
deprecated, use PAE_Analytical instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kCSD_Numerical(PAE_Numerical):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_Numerical class is \
deprecated, use PAE_Numerical instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kCSD_AnalyticalMasked(PAE_AnalyticalMaskedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_AnalyticalMasked class is \
    deprecated, use PAE_AnalyticalMaskedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kCSD_NumericalMasked(PAE_NumericalMasked):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_NumericalMasked class is \
deprecated, use PAE_NumericalMasked instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kESI_AnalyticalMasked(PAE_AnalyticalMaskedAndCorrectedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kESI_AnalyticalMasked class is \
    deprecated, use PAE_AnalyticalMaskedAndCorrectedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kESI_Analytical(PAE_AnalyticalCorrectedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kESI_Analytical class is \
    deprecated, use PAE_AnalyticalCorrectedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    import itertools
    import _common_new as common

    ns = [6 * i for i in range(1, 4)]
    pot_grid = [np.linspace(-1, 1, n // 3 + 1) for n in ns]
    csd_grid = [np.linspace(-1, 1, n // 2 + 1) for n in ns]
    src_grid = [np.linspace(-1, 1, n // 6 + 1) for n in ns]

    conv = Convolver(pot_grid, csd_grid)

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
                CSD = conv.base_weights_to_csd(SRC, csd, [3, 3, 5])
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

    POT = conv.leadfield_to_base_potentials(LEADFIELD,
                                            lambda x, y, z: 1 / acc,
                                            [[0.5, 0, 0.5]] * 3)

    for (wx, idx_x), (wy, idx_y), (wz, idx_z) in itertools.product([(0.5, 0),
                                                                    (1, slice(1,
                                                                              -1)),
                                                                    (0.5, -1)],
                                                                   repeat=3):
        assert (abs(POT[idx_x, idx_y, idx_z] - wx * wy * wz) < 1e-11).all()

    grid = [np.linspace(-1.1, -1, 100)] * 3
    conv = Convolver(grid, grid)

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
    POT = conv.leadfield_to_base_potentials(LEADFIELD, model_src.csd,
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

    convolver = Convolver([X, Y, Z], [X, Y, Z])
    romberg_weights = tuple(si.romb(np.identity(2 ** _k + 1)) / 2 ** _k
                            for _k in range(ROMBERG_K, ROMBERG_K + 3))

    SRC_MASK = ((convolver.SRC_X >= 2 * R) & (convolver.SRC_X <= 8 * R)) & (
                (convolver.SRC_Y >= -0.5 * R) & (
                    convolver.SRC_Y <= 1.5 * R)) & (
                     (convolver.SRC_Z >= R) & (convolver.SRC_Z <= 3 * R))

    convolver_interface = ConvolverInterfaceIndexed(convolver,
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

    tested = PAE_Analytical(convolver_interface,
                            potential=model_src.potential)
    with tested:
        observed = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-10, ECHO)

    # kCSD numeric
    tested = PAE_Numerical(convolver_interface)
    with tested:
        observed = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-2, ECHO)

    # kCSD masked
    # kCSD masked analytical
    tested = PAE_AnalyticalMaskedNumerically(convolver_interface,
                                             potential=model_src.potential,
                                             leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kcsd)
    tested = PAE_AnalyticalMaskedNumerically(convolver_interface,
                                             potential=model_src.potential,
                                             leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kcsd)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)

    # kCSD masked numerical
    tested = PAE_NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kcsd)
    tested = PAE_NumericalMasked(convolver_interface,
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
    tested = PAE_NumericalCorrection(convolver_interface)
    with tested:
        observed = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(correction, observed, 2e-4, ECHO)


    expected += correction

    # kESI analytical
    tested = PAE_AnalyticalCorrectedNumerically(convolver_interface,
                                                potential=model_src.potential)
    with tested:
        observed = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-4, ECHO)

    # kESI numerical
    tested = PAE_Numerical(convolver_interface)
    with tested:
        observed = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected, observed, 1e-2, ECHO)

    # kESI analytical masked
    tested = PAE_AnalyticalMaskedAndCorrectedNumerically(convolver_interface,
                                                         potential=model_src.potential,
                                                         leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kesi)
    tested = PAE_AnalyticalMaskedAndCorrectedNumerically(convolver_interface,
                                                         potential=model_src.potential,
                                                         leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)

    # kESI numerical masked
    tested = PAE_NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MAJOR)
    with tested:
        observed_major = tested(test_electrode_kesi)
    tested = PAE_NumericalMasked(convolver_interface,
                                 leadfield_allowed_mask=MASK_MINOR)
    with tested:
        observed_minor = tested(test_electrode_kesi)
    assertRelativeErrorWithinTolerance(expected,
                                       (observed_major + observed_minor),
                                       1e-2,
                                       ECHO)
