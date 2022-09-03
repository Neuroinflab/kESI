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


def shape(axis, n=3):
    return [-1 if i == axis else 1
            for i in range(n)]


def reshape(A, axis, n=3):
    return np.reshape(A, shape(axis, n))


class ckESI_convolver(object):
    def __init__(self, potential_mesh,
                 csd_mesh):
        self.POT_MESH = []
        self.CSD_MESH = []
        self.SRC_MESH = []

        for i, (c, POT, CSD) in enumerate(zip(['X', 'Y', 'Z'],
                                              potential_mesh,
                                              csd_mesh)):
            POT = reshape(POT, i)
            CSD = reshape(CSD, i)

            SRC, CSD_IDX, POT_IDX = np.intersect1d(
                CSD.flatten(),
                POT.flatten(),
                assume_unique=True,
                return_indices=True)
            SRC = reshape(SRC, i)

            self.POT_MESH.append(POT)
            self.CSD_MESH.append(CSD)
            self.SRC_MESH.append(SRC)

            setattr(self, f'CSD_{c}', CSD)
            setattr(self, f'POT_{c}', POT)
            setattr(self, f'SRC_{c}', SRC)
            setattr(self, f'_SRC_CSD_IDX_{c}', CSD_IDX)
            setattr(self, f'_SRC_POT_IDX_{c}', POT_IDX)

    def leadfield_to_base_potentials(self,
                                     LEADFIELD,
                                     csd,
                                     weights):

        ds = self.ds('POT')

        ns = list(map(len, weights))

        CSD = self.csd_kernel(csd, ns, ds)

        WEIGHTS = functools.reduce(np.multiply,
                                   [d * (n - 1) * reshape(w, i)
                                    for i, (w, n, d) in enumerate(zip(weights,
                                                                      ns,
                                                                      ds))])
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
                               self.csd_kernel(csd, csd_ns, self.ds('CSD')),
                               mode='same')

    def csd_kernel(self, csd, ns, ds):
        return csd(*np.meshgrid(*[np.array([0.]) if np.isnan(d) else
                                  d * np.linspace(-(n // 2), n // 2, n)
                                  for d, n in zip(ds, ns)],
                                indexing='ij',
                                sparse=True))

    def ds(self, name):
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
                     for i, S in enumerate(getattr(self, f'{name}_MESH')))


class ckESI_kernel_constructor_no_cross(object):
    def __init__(self,
                 model_source,
                 convolver,
                 source_indices,
                 electrodes,
                 weights=65,
                 leadfield_allowed_mask=None,
                 source_normalization_treshold=None):
        if isinstance(weights, int):
            self._src_diameter = weights
            weights = si.romb(np.identity(weights)) / (weights - 1)
        else:
            self._src_diameter = len(weights)

        self.convolver = convolver
        self.source_indices = source_indices
        self.model_source = model_source
        self.leadfield_allowed_mask = leadfield_allowed_mask
        self.source_normalization_treshold = source_normalization_treshold

        self._create_pre_kernel(electrodes, weights)
        self._create_kernel()

    def _create_pre_kernel(self, electrodes, weights):
        kcsd_solution_available = hasattr(self.model_source, 'potential')

        if kcsd_solution_available:
            SRC_X, SRC_Y, SRC_Z = np.meshgrid(self.convolver.SRC_X,
                                              self.convolver.SRC_Y,
                                              self.convolver.SRC_Z,
                                              indexing='ij')
            SRC_X = SRC_X[self.source_indices]
            SRC_Y = SRC_Y[self.source_indices]
            SRC_Z = SRC_Z[self.source_indices]

        if (not kcsd_solution_available
            or any(hasattr(e, 'correction_potential')
                   for e in electrodes)):
            POT_XYZ = np.meshgrid(self.convolver.POT_X,
                                  self.convolver.POT_Y,
                                  self.convolver.POT_Z,
                                  indexing='ij')
            if self.leadfield_allowed_mask is not None:
                # XXX Applies CSD mask to potential XYZs ?
                POT_XYZ_MASKED = [A[self.leadfield_allowed_mask]
                                  for A in POT_XYZ]

        if kcsd_solution_available and self.leadfield_allowed_mask is not None:
            CSD_FORBIDDEN_MASK = ~self.leadfield_allowed_mask
            # XXX Applies CSD mask to potential XYZs ?
            POT_XYZ_CROPPED = [A[CSD_FORBIDDEN_MASK]
                               for A in POT_XYZ]

        LEADFIELD = None
        for i, electrode in enumerate(electrodes):
            correction_available = hasattr(electrode, 'correction_potential')

            leadfield_updated = not (kcsd_solution_available
                                     and self.leadfield_allowed_mask is None
                                     and not correction_available)

            if self.leadfield_allowed_mask is not None:
                if correction_available and kcsd_solution_available:
                    LEADFIELD = self.alloc_leadfield_if_necessary(LEADFIELD)
                else:
                    LEADFIELD = self.clear_leadfield(LEADFIELD)

                if kcsd_solution_available:
                    # XXX Applies CSD mask to potential ?
                    LEADFIELD[CSD_FORBIDDEN_MASK] = -electrode.base_potential(*POT_XYZ_CROPPED)
                    if correction_available:
                        LEADFIELD[self.leadfield_allowed_mask] = electrode.correction_potential(*POT_XYZ)[self.leadfield_allowed_mask]
                else:
                    # XXX Applies CSD mask to potential ?
                    LEADFIELD[self.leadfield_allowed_mask] = electrode.base_potential(*POT_XYZ_MASKED)
                    if correction_available:
                        LEADFIELD[self.leadfield_allowed_mask] += electrode.correction_potential(*POT_XYZ)[self.leadfield_allowed_mask]

            else:
                if kcsd_solution_available:
                    if correction_available:
                        LEADFIELD = electrode.correction_potential(*POT_XYZ)
                else:
                    LEADFIELD = electrode.base_potential(*POT_XYZ)
                    if correction_available:
                        LEADFIELD += electrode.correction_potential(*POT_XYZ)

            if kcsd_solution_available:
                POT = self.model_source.potential(electrode.x - SRC_X,
                                                  electrode.y - SRC_Y,
                                                  electrode.z - SRC_Z)
            else:
                POT = 0

            if leadfield_updated:
                POT += self.integrate_source_potential(LEADFIELD, weights)

            if i == 0:
                n_bases = POT.size
                self._pre_kernel = np.full((n_bases, len(electrodes)),
                                           np.nan)

            self._pre_kernel[:, i] = POT
        self._pre_kernel /= n_bases

        if self.normalize_sources():
            self.calculate_source_normalization_factor(weights)
            self._pre_kernel *= self.source_normalization_factor.reshape(-1, 1)

    def calculate_source_normalization_factor(self, weights):
        current = self.integrate_source_potential(
            self.leadfield_allowed_mask,
            weights)
        self.source_normalization_factor = 1.0 / np.where(abs(current) > self.source_normalization_treshold,
                                                          current,
                                                          self.source_normalization_treshold)

    def normalize_sources(self):
        return (self.leadfield_allowed_mask is not None
                and self.source_normalization_treshold is not None)

    def alloc_leadfield_if_necessary(self, leadfield):
        if leadfield is not None:
            return leadfield
        return np.empty(self.convolver.shape('POT'))

    def clear_leadfield(self, leadfield):
        if leadfield is None:
            return np.zeros(self.convolver.shape('POT'))

        leadfield.fill(0)
        return leadfield

    def integrate_source_potential(self, leadfield, quadrature_weights):
        return self.convolve_csd(leadfield,
                                 quadrature_weights)[self.source_indices]

    def convolve_csd(self, leadfield, quadrature_weights):
        return self.convolver.leadfield_to_base_potentials(
            leadfield,
            self.model_source.csd,
            [quadrature_weights] * 3)

    def _create_kernel(self):
        self.kernel = np.matmul(self._pre_kernel.T,
                                self._pre_kernel) * len(self._pre_kernel)


class ckESI_kernel_constructor(object):
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


class ckESI_crosskernel_constructor(object):
    def __init__(self,
                 convolver_interface,
                 csd_indices,
                 csd_allowed_mask=None):
        self.ci = convolver_interface
        self.csd_indices = csd_indices
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
        return csd[self.csd_indices]

    def _allocate_cross_kernel(self, n_points):
        self._cross_kernel = np.full((n_points, self._n_electrodes),
                                     np.nan)

    def _zero_crosskernel_where_csd_not_allowed(self):
        if self.csd_allowed_mask is not None:
            self._cross_kernel[~self._crop_csd(self.csd_allowed_mask), :] = 0


class ConvolverInterface_base(object):
    def __init__(self, convolver, csd, weights):
        self.convolver = convolver
        self.csd = csd
        self.weights = weights

    @property
    def _src_diameter(self):
        return len(self.weights)

    def convolve_csd(self, leadfield):
        return self.convolver.leadfield_to_base_potentials(
            leadfield,
            self.csd,
            [self.weights] * 3)

    def zeros(self, name):
        return np.zeros(self.convolver.shape(name))

    def empty(self, name):
        return np.empty(self.convolver.shape(name))

    def base_weights_to_csd(self, base_weights):
        csd_kernel_shape = [(1 if np.isnan(csd)
                             else int(round(self._src_diameter * pot / csd) - 1))
                            for pot, csd in zip(*map(self.convolver.ds,
                                                     ['POT', 'CSD']))]
        return self.convolver.base_weights_to_csd(base_weights,
                                                  self.csd,
                                                  csd_kernel_shape)

    def meshgrid(self, name):
        return np.meshgrid(*getattr(self.convolver,
                                    f'{name}_MESH'),
                           indexing='ij')


class ConvolverInterfaceIndexed(ConvolverInterface_base):
    def __init__(self, convolver, csd, weights, source_indices):
        super().__init__(convolver, csd, weights)
        self.source_indices = source_indices

    def integrate_source_potential(self, leadfield):
        return self.convolve_csd(leadfield)[self.source_indices]

    def src_coords(self):
        return [A[self.source_indices] for A in self.meshgrid('SRC')]

    def update_src(self, src, values):
        src[self.source_indices] = values


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

class PAE_kCSD_Analytical(PAE_Analytical):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_Analytical class is \
deprecated, use PAE_Analytical instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_Numerical(_PAE_LeadfieldFromElectrode,
                    _PAE_PotAttribute):
    pass


class PAE_kCSD_Numerical(PAE_Numerical):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_Numerical class is \
deprecated, use PAE_Numerical instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_AnalyticalMaskedNumerically(_PAE_PotProperty,
                                      _PAE_LeadfieldCroppingAnalyticalBasesNumerically,
                                      PAE_Analytical):
    _LEADFIELD_METHOD = 'leadfield'


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

# kESI

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


class PAE_kESI_AnalyticalMasked(PAE_AnalyticalMaskedAndCorrectedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kESI_AnalyticalMasked class is \
    deprecated, use PAE_AnalyticalMaskedAndCorrectedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class _PAE_LeadfieldFromCorrectionPotential(_PAE_FromLeadfieldNotMasked,
                                            _PAE_PotAttribute):
    def _create_leadfield(self, electrode):
        super()._create_leadfield(electrode)
        LEADFIELD = electrode.correction_leadfield(*self.POT_XYZ)
        if self.LEADFIELD is not None:
            self.LEADFIELD += LEADFIELD
        else:
            self.LEADFIELD = LEADFIELD


class PAE_AnalyticalCorrectedNumerically(_PAE_LeadfieldFromCorrectionPotential,
                                         PAE_Analytical):
    pass


class PAE_kESI_Analytical(PAE_AnalyticalCorrectedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kESI_Analytical class is \
    deprecated, use PAE_AnalyticalCorrectedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


# class ckCSD_kernel_constructor_MOI(ckESI_kernel_constructor):
#     def __init__(self,
#                  model_source,
#                  convolver,
#                  source_indices,
#                  csd_indices,
#                  electrodes,
#                  slice_thickness,
#                  slice_conductivity,
#                  saline_conductivity,
#                  glass_conductivity=0.0,
#                  n=128
#                  ):
#         self.slice_thickness = slice_thickness
#         self.slice_conductivity = slice_conductivity
#         self.saline_conductivity = saline_conductivity
#         self.glass_conductivity = glass_conductivity
#         self.n = n
#         super(ckCSD_kernel_constructor_MOI,
#               self).__init__(model_source,
#                              convolver,
#                              source_indices,
#                              csd_indices,
#                              electrodes)
#
#     def _create_pre_kernel(self, electrodes, weights):
#         wtg = float(self.slice_conductivity - self.glass_conductivity) / (
#                     self.slice_conductivity + self.glass_conductivity)
#         wts = float(self.slice_conductivity - self.saline_conductivity) / (
#                     self.slice_conductivity + self.saline_conductivity)
#
#         weights = [1.0]
#         for i in range(self.n):
#             weights.append(wtg ** i * wts ** (i + 1))
#             weights.append(wtg ** (i + 1) * wts ** i)
#
#         for i in range(1, self.n + 1):
#             weights.append((wtg * wts) ** i)
#             weights.append((wtg * wts) ** i)
#
#         weights = np.array(weights)
#
#         SRC_X, SRC_Y, SRC_Z = np.meshgrid(self.ci.convolver.SRC_X,
#                                           self.ci.convolver.SRC_Y,
#                                           self.ci.convolver.SRC_Z,
#                                           indexing='ij')
#         SRC_X = SRC_X[self.source_indices].reshape(-1, 1)
#         SRC_Y = SRC_Y[self.source_indices].reshape(-1, 1)
#         SRC_Z = SRC_Z[self.source_indices].reshape(-1, 1)
#         n_bases = SRC_X.size
#
#         self._pre_kernel = np.full((n_bases, len(electrodes)),
#                                    np.nan)
#         for i_ele, electrode in enumerate(electrodes):
#             ele_z = [electrode.z]
#             for i in range(self.n):
#                 ele_z.append(2 * (i + 1) * self.slice_thickness - electrode.z)
#                 ele_z.append(-2 * i * self.slice_thickness - electrode.z)
#
#             for i in range(1, self.n + 1):
#                 ele_z.append(electrode.z + 2 * i * self.slice_thickness)
#                 ele_z.append(electrode.z - 2 * i * self.slice_thickness)
#
#             ele_z = np.reshape(ele_z, (1, -1))
#
#             POT = self.model_source.potential(electrode.x - SRC_X,
#                                               electrode.y - SRC_Y,
#                                               ele_z - SRC_Z)
#
#             self._pre_kernel[:, i_ele] = np.matmul(POT, weights)
#         self._pre_kernel /= n_bases


if __name__ == '__main__':
    import itertools
    import _common_new as common

    ns = [6 * i for i in range(1, 4)]
    pot_mesh = [np.linspace(-1, 1, n // 3 + 1) for n in ns]
    csd_mesh = [np.linspace(-1, 1, n // 2 + 1) for n in ns]
    src_mesh = [np.linspace(-1, 1, n // 6 + 1) for n in ns]

    conv = ckESI_convolver(pot_mesh, csd_mesh)

    for name, expected_mesh in [('POT', pot_mesh),
                                ('CSD', csd_mesh),
                                ('SRC', src_mesh)]:
        observed_mesh = getattr(conv, f'{name}_MESH')

        for i, (coord, expected) in enumerate(zip(['X', 'Y', 'Z'],
                                                  expected_mesh)):
            observed = getattr(conv, f'{name}_{coord}')

            assert (observed == observed_mesh[i]).all()
            assert observed.shape == observed_mesh[i].shape

            assert (observed.flatten() == expected).all()
            for j, n in enumerate(observed.shape):
                assert n == (1 if j != i else len(expected))


    def csd(x, y, z):
        return np.maximum(1 - (x ** 2 + y ** 2 + z ** 2) * 4, 0)


    for i_x in range(conv.SRC_X.shape[0]):
        for i_y in range(conv.SRC_Y.shape[1]):
            for i_z in range(conv.SRC_Z.shape[2]):
                SRC = np.zeros(
                    [S.shape[i] for i, S in enumerate(conv.SRC_MESH)])
                SRC[i_x, i_y, i_z] = 2
                CSD = conv.base_weights_to_csd(SRC, csd, [3, 3, 5])
                EXP = 2 * csd(conv.CSD_X - conv.SRC_X[i_x, 0, 0],
                              conv.CSD_Y - conv.SRC_Y[0, i_y, 0],
                              conv.CSD_Z - conv.SRC_Z[0, 0, i_z])
                ERR = CSD - EXP
                assert abs(ERR).max() < 1e-11

    LEADFIELD = np.ones([S.shape[i] for i, S in enumerate(conv.POT_MESH)])
    LEADFIELD[conv.src_idx('POT')] -= 1
    acc = 8
    for d in conv.ds('POT'):
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

    mesh = [np.linspace(-1.1, -1, 100)] * 3
    conv = ckESI_convolver(mesh, mesh)

    conductivity = 0.33
    LEADFIELD = 0.25 / np.pi / conductivity / np.sqrt(np.square(conv.POT_X)
                                                      + np.square(conv.POT_Y)
                                                      + np.square(conv.POT_Z))
    WEIGHTS = si.romb(np.identity(65)) / 64

    sd = conv.ds('POT')[0] * 64 / 6
    model_src = common.SphericalSplineSourceKCSD(0, 0, 0,
                                                 [sd, 3 * sd],
                                                 [[1],
                                                  [0,
                                                   2.25 / sd,
                                                   -1.5 / sd ** 2,
                                                   0.25 / sd ** 3]],
                                                 conductivity)

    POT_GT = model_src.potential(*conv.SRC_MESH)
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
    Y = np.linspace(-1.5 * R, 2.5 * R, 2 ** (ROMBERG_K + 1) + 1)
    Z = np.linspace(0, 4 * R, 2 ** (ROMBERG_K + 1) + 1)

    convolver = ckESI_convolver([X, Y, Z], [X, Y, Z])
    romberg_weights = si.romb(np.identity(ROMBERG_N)) / (ROMBERG_N - 1)

    SRC_IDX = ((convolver.SRC_X >= 2 * R) & (convolver.SRC_X <= 8 * R)) & (
                (convolver.SRC_Y >= -0.5 * R) & (
                    convolver.SRC_Y <= 1.5 * R)) & (
                     (convolver.SRC_Z >= R) & (convolver.SRC_Z <= 3 * R))

    convolver_interface = ConvolverInterfaceIndexed(convolver,
                                                    model_src.csd,
                                                    romberg_weights,
                                                    SRC_IDX)


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
                                        convolver.SRC_Z)[SRC_IDX]
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
    expected += reciprocal_src.potential(-convolver.SRC_X,
                                         convolver.SRC_Y,
                                         convolver.SRC_Z)[SRC_IDX]

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
