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

import functools
import warnings

import numpy as np
from scipy import signal as ssi

from ._tools import reshape


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

    def leadfield_to_potential_basis_functions(self,
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
        # import IPython
        # IPython.embed()
        # import pylab as pb
        # globals().update(locals())
        # def visualise(LEADFIELD):
        #     slices = 30
        #     fig, axs = pb.subplots(slices, 1, figsize=(5, 5*slices))
        #     for i in range(slices):
        #         z = int(LEADFIELD.shape[-1] / slices * i)
        #         axs[i].imshow(LEADFIELD[:, :, z])
        #     fig.tight_layout()
        #     fig.savefig("test.png")
        #     pb.show()
        # visualise(LEADFIELD)
        return ssi.fftconvolve(LEADFIELD,
                               CSD * WEIGHTS,
                               mode='same')[self.src_idx('POT')]

    def basis_functions_weights_to_csd(self, BASE_WEIGHTS, csd, csd_ns):
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


class KernelConstructor(object):
    class NoElectrodesGivenException(ValueError):
        @classmethod
        def check(cls, electrodes):
            if len(electrodes) == 0:
                raise cls

    @staticmethod
    def kernel(potential_basis_functions_at_electrodes):
        return np.matmul(potential_basis_functions_at_electrodes.T,
                         potential_basis_functions_at_electrodes)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._potential_basis_functions

    def potential_basis_functions_at_electrodes(self,
                                                electrodes,
                                                potential_basis_functions):
        self.NoElectrodesGivenException.check(electrodes)

        with self:
            with potential_basis_functions:
                self._calculate_potential_basis_functions_at_electrodes(
                                                      electrodes,
                                                      potential_basis_functions)

            return self._potential_basis_functions

    def _calculate_potential_basis_functions_at_electrodes(self,
                                                     electrodes,
                                                     potential_basis_functions):
        for i, electrode in enumerate(electrodes):
            POT = potential_basis_functions(electrode)

            self._alloc_potential_basis_functions_if_necessary(POT.size,
                                                               len(electrodes))
            self._potential_basis_functions[:, i] = POT

    def _alloc_potential_basis_functions_if_necessary(self,
                                                      n_bases,
                                                      n_electrodes):
        if not hasattr(self, '_potential_basis_functions'):
            self._potential_basis_functions = np.full((n_bases, n_electrodes),
                                                      np.nan)


class CrossKernelConstructor(object):
    def __init__(self,
                 convolver_interface,
                 csd_mask,
                 csd_allowed_mask=None):
        self.ci = convolver_interface
        self.csd_mask = csd_mask
        self.csd_allowed_mask = csd_allowed_mask

    def __enter__(self):
        self._basis_functions_weights = self.ci.zeros('SRC')

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._basis_functions_weights
        for attr in ['_n_electrodes', '_cross_kernel']:
            if hasattr(self, attr):
                delattr(self, attr)

    def __call__(self, potential_basis_functions_at_electrodes):
        with self:
            self._create_crosskernel(potential_basis_functions_at_electrodes)
            return self._cross_kernel

    def _create_crosskernel(self, potential_basis_functions_at_electrodes):
        self._n_electrodes = potential_basis_functions_at_electrodes.shape[1]
        for i, potential_basis_functions in enumerate(
                                     potential_basis_functions_at_electrodes.T):
            self.ci.update_src(self._basis_functions_weights,
                               potential_basis_functions)
            self._set_crosskernel_column(i, self._basis_functions_to_csd())

        self._zero_crosskernel_where_csd_not_allowed()

    def _set_crosskernel_column(self, i, column):
        if i == 0:
            self._allocate_cross_kernel(column.size)

        self._cross_kernel[:, i] = column

    def _basis_functions_to_csd(self):
        return self._crop_csd(self.ci.basis_functions_weights_to_csd(
                                                 self._basis_functions_weights))

    def _crop_csd(self, csd):
        return csd[self.csd_mask]

    def _allocate_cross_kernel(self, n_points):
        self._cross_kernel = np.full((n_points, self._n_electrodes),
                                     np.nan)

    def _zero_crosskernel_where_csd_not_allowed(self):
        if self.csd_allowed_mask is not None:
            self._cross_kernel[~self._crop_csd(self.csd_allowed_mask), :] = 0


class ConvolverInterface_base(object):
    """
    Parameters
    ----------
    weights : sequence or tuple of sequences
        weights of the quadrature

    Note
    ----
    If `weights` are tuple they are interpreted as weights of quadrature in X, Y
    and Z direction.
    """
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
        return self.convolver.leadfield_to_potential_basis_functions(
            leadfield,
            self.csd,
            list(self.weights))

    def zeros(self, name):
        return np.zeros(self.convolver.shape(name))

    def empty(self, name):
        return np.empty(self.convolver.shape(name))

    def basis_functions_weights_to_csd(self, basis_functions_weights):
        csd_kernel_shape = [(1 if np.isnan(csd)
                             else int(round(r * pot / csd)) * 2 + 1)
                            for r, pot, csd in zip(self._src_radius,
                                                   *map(self.convolver.steps,
                                                     ['POT', 'CSD']))]
        return self.convolver.basis_functions_weights_to_csd(
                                                        basis_functions_weights,
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
