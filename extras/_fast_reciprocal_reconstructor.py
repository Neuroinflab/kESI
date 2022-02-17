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
import numpy as np
import scipy.integrate as si
import scipy.signal as ssi

import kesi

try:
    from . import _common_new as common
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import _common_new as common


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


class ckESI_kernel_constructor_base(object):
    def __init__(self,
                 model_source,
                 convolver,
                 source_indices,
                 csd_indices,
                 electrodes,
                 weights=65,
                 leadfield_allowed_mask=None,
                 source_normalization_treshold=None,
                 csd_allowed_mask=None):
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
        self.csd_indices = csd_indices
        self.csd_allowed_mask = csd_allowed_mask

        self._create_pre_kernel(electrodes,
                                self._potential_at_electrode(
                                    leadfield_allowed_mask,
                                    weights))
        self._normalize_pre_kernel(weights)
        self._create_kernel()
        self._create_crosskernel()

    def calculate_source_normalization_factor(self, weights):
        current = self.integrate_source_potential(
            self.leadfield_allowed_mask,
            weights)
        self.source_normalization_factor = 1.0 / np.where(abs(current) > self.source_normalization_treshold,
                                                          current,
                                                          self.source_normalization_treshold)

    def source_normalization_requested(self):
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

    def _create_crosskernel(self):
        SRC = np.zeros(self.convolver.shape('SRC'))
        for i, PHI_COL in enumerate(self._pre_kernel.T):
            SRC[self.source_indices] = (PHI_COL * self.source_normalization_factor
                                        if self.source_normalization_requested()
                                        else PHI_COL)
            CROSS_COL = self._base_weights_to_csd(SRC)
            if i == 0:
                self._allocate_cross_kernel(CROSS_COL)

            self.cross_kernel[:, i] = CROSS_COL

        self._zero_cross_kernel_where_csd_not_allowed()

    def _allocate_cross_kernel(self, CROSS_COL):
        self.cross_kernel = np.full((CROSS_COL.size,
                                     self._pre_kernel.shape[1]),
                                    np.nan)

    def _base_weights_to_csd(self, BASE_WEIGHTS):
        csd_kernel_shape = [(1 if np.isnan(csd)
                             else int(round(self._src_diameter * pot / csd) - 1))
                            for pot, csd in zip(*map(self.convolver.ds,
                                                     ['POT', 'CSD']))]
        return self.convolver.base_weights_to_csd(BASE_WEIGHTS,
                                                  self.model_source.csd,
                                                  csd_kernel_shape)[self.csd_indices]

    def _zero_cross_kernel_where_csd_not_allowed(self):
        if self.csd_allowed_mask is not None:
            self.cross_kernel[~self.csd_allowed_mask[self.csd_indices], :] = 0

    def _normalize_pre_kernel(self, weights):
        self._pre_kernel /= self._pre_kernel.shape[0]
        if self.source_normalization_requested():
            self.calculate_source_normalization_factor(weights)
            self._pre_kernel *= self.source_normalization_factor.reshape(-1, 1)

    def _create_pre_kernel(self, electrodes, potential_at_electrode):
        for i, electrode in enumerate(electrodes):
            POT = potential_at_electrode(electrode)

            self._alloc_pre_kernel_if_necessary(POT.size, len(electrodes))
            self._pre_kernel[:, i] = POT

    def _alloc_pre_kernel_if_necessary(self, n_bases, n_electrodes):
        if not hasattr(self, '_pre_kernel'):
            self._pre_kernel = np.full((n_bases, n_electrodes),
                                       np.nan)

    def _potential_at_electrode(self, leadfield_allowed_mask, weights):
        if leadfield_allowed_mask is None:
            return self._potential_at_electrode_simple(weights)

        return self._potential_at_electrode_masked(weights, leadfield_allowed_mask)

    def _potential_at_electrode_simple(self, weights):
        if self._kcsd_solution_available():
            return self._PotentialAtElectrodeAnalytical(self, weights)

        return self._PotentialAtElectrodeNumerical(self, weights)

    def _potential_at_electrode_masked(self, weights, leadfield_allowed_mask):
        if self._kcsd_solution_available():
            return self._PotentialAtElectrodeAnalyticalMasked(self,
                                                              weights,
                                                              leadfield_allowed_mask)

        return self._PotentialAtElectrodeNumericalMasked(self,
                                                         weights,
                                                         leadfield_allowed_mask)

    def _kcsd_solution_available(self):
        return hasattr(self.model_source, 'potential')

    class _PotentialAtElectrodeBase(object):
        def __init__(self, parent, weights):
            self.parent = parent
            self.weights = weights

        def __call__(self, electrode):
            return None

        @property
        def convolver(self):
            return self.parent.convolver

        @property
        def source_indices(self):
            return self.parent.source_indices

        @property
        def model_source(self):
            return self.parent.model_source

    class _PotentialAtElectrodeAnalytical(_PotentialAtElectrodeBase):
        def __init__(self, parent, weights):
            super().__init__(parent, weights)
            SRC_X, SRC_Y, SRC_Z = np.meshgrid(self.convolver.SRC_X,
                                              self.convolver.SRC_Y,
                                              self.convolver.SRC_Z,
                                              indexing='ij')
            self.SRC_X = SRC_X[self.source_indices]
            self.SRC_Y = SRC_Y[self.source_indices]
            self.SRC_Z = SRC_Z[self.source_indices]

        def __call__(self, electrode):
            V = self.model_source.potential(electrode.x - self.SRC_X,
                                            electrode.y - self.SRC_Y,
                                            electrode.z - self.SRC_Z)
            V_SUPER = super().__call__(electrode)
            if V_SUPER is not None:
                V += V_SUPER

            return V

    class _PotentialAtElectrodePotAttribute(_PotentialAtElectrodeBase):
        def __init__(self, parent, weights):
            super().__init__(parent, weights)
            self.POT_XYZ = np.meshgrid(self.convolver.POT_X,
                                       self.convolver.POT_Y,
                                       self.convolver.POT_Z,
                                       indexing='ij')

    class _PotentialAtElectrodeFromLeadfield(_PotentialAtElectrodeBase):
        def __call__(self, electrode):
            self._create_leadfield(electrode)
            V = self.integrate_source_potential()
            V_SUPER = super().__call__(electrode)
            if V_SUPER is not None:
                V += V_SUPER

            return V

        def integrate_source_potential(self):
            return self.convolve_csd()[self.source_indices]

        def convolve_csd(self):
            return self.convolver.leadfield_to_base_potentials(
                self.LEADFIELD,
                self.model_source.csd,
                [self.weights] * 3)

    class _PotentialAtElectrodeMasked(_PotentialAtElectrodeFromLeadfield):
        def __init__(self, parent, weights, leadfield_allowed_mask):
            super().__init__(parent, weights)
            self.leadfield_allowed_mask = leadfield_allowed_mask

        def _create_leadfield(self, electrode):
            self._provide_leadfield_array()

        def _provide_leadfield_array(self):
            self.clear_leadfield()

        def clear_leadfield(self):
            try:
                self.LEADFIELD.fill(0)

            except AttributeError:
                self.LEADFIELD = np.zeros(self.convolver.shape('POT'))

    class _PotentialAtElectrodeLeadfieldForbiddenMask(
            _PotentialAtElectrodeMasked):
        """
        `.POT_XYZ` attribute/property required
        """
        def __init__(self, parent, weights, leadfield_allowed_mask):
            super().__init__(parent, weights, leadfield_allowed_mask)

            self.csd_forbidden_mask = ~leadfield_allowed_mask
            self.POT_XYZ_CROPPED = [A[self.csd_forbidden_mask] for A in self.POT_XYZ]

        def _create_leadfield(self, electrode):
            super()._create_leadfield(electrode)
            self.LEADFIELD[self.csd_forbidden_mask] = -electrode.base_potential(*self.POT_XYZ_CROPPED)

    class _PotentialAtElectrodeNumericalMask(_PotentialAtElectrodeMasked):
        """
        `.POT_XYZ` attribute/property required
        """
        def __init__(self, parent, weights, leadfield_allowed_mask):
            super().__init__(parent, weights, leadfield_allowed_mask)

            self.POT_XYZ_MASKED = [A[leadfield_allowed_mask] for A in self.POT_XYZ]

        def _create_leadfield(self, electrode):
            super()._create_leadfield(electrode)
            self.LEADFIELD[self.leadfield_allowed_mask] += electrode.base_potential(*self.POT_XYZ_MASKED)

    class _PotentialAtElectrodeFromLeadfieldNotMasked(_PotentialAtElectrodeFromLeadfield):
        def _create_leadfield(self, electrode):
            self.LEADFIELD = None

    class _PotentialAtElectrodeNumerical(_PotentialAtElectrodeFromLeadfieldNotMasked):
        """
        `.POT_XYZ` attribute required
        """
        def _create_leadfield(self, electrode):
            super()._create_leadfield(electrode)
            LEADFIELD = electrode.base_potential(*self.POT_XYZ)
            if self.LEADFIELD is not None:
                self.LEADFIELD += LEADFIELD
            else:
                self.LEADFIELD = LEADFIELD


class ckESI_kernel_constructor(ckESI_kernel_constructor_base):
    class _PotentialAtElectrode(
            ckESI_kernel_constructor_base._PotentialAtElectrodeFromLeadfield,
            ckESI_kernel_constructor_base._PotentialAtElectrodePotAttribute):
        pass

    class _PotentialAtElectrodeMasked(
            ckESI_kernel_constructor_base._PotentialAtElectrodeMasked,
            _PotentialAtElectrode):
        def _create_leadfield(self, electrode):
            super()._create_leadfield(electrode)
            # `.correction_potential(XS)[IDX]` used instead of
            # `.correction_potential(XS[IDX]) to simplify implementation
            # of the method
            CORRECTION = electrode.correction_potential(*self.POT_XYZ)[self.leadfield_allowed_mask]
            self.LEADFIELD[self.leadfield_allowed_mask] = CORRECTION

    class _PotentialAtElectrodeAnalyticalMasked(
            ckESI_kernel_constructor_base._PotentialAtElectrodeLeadfieldForbiddenMask,
            _PotentialAtElectrodeMasked,
            ckESI_kernel_constructor_base._PotentialAtElectrodeAnalytical):
        def _provide_leadfield_array(self):
            self.alloc_leadfield_if_necessary()

        def alloc_leadfield_if_necessary(self):
            if not hasattr(self, 'LEADFIELD'):
                self.LEADFIELD = np.empty(self.convolver.shape('POT'))

    class _PotentialAtElectrodeNumericalMasked(
        ckESI_kernel_constructor_base._PotentialAtElectrodeNumericalMask,
        _PotentialAtElectrodeMasked):
        # MRO counts - it is crucial to finish call to
        # `_PotentialAtElectrodeMasked._create_leadfield()` (assign) before
        # `_PotentialAtElectrodeNumericalMask._create_leadfield()` (add)
        pass

    class _PotentialAtElectrodeNotMasked(
            ckESI_kernel_constructor_base._PotentialAtElectrodeFromLeadfieldNotMasked,
            _PotentialAtElectrode):
        def _create_leadfield(self, electrode):
            super()._create_leadfield(electrode)
            LEADFIELD = electrode.correction_potential(*self.POT_XYZ)
            if self.LEADFIELD is not None:
                self.LEADFIELD += LEADFIELD
            else:
                self.LEADFIELD = LEADFIELD

    class _PotentialAtElectrodeAnalytical(
            _PotentialAtElectrodeNotMasked,
            ckESI_kernel_constructor_base._PotentialAtElectrodeAnalytical):
        pass

    class _PotentialAtElectrodeNumerical(
            _PotentialAtElectrodeNotMasked,
            ckESI_kernel_constructor_base._PotentialAtElectrodeNumerical):
        pass


class ckCSD_kernel_constructor(ckESI_kernel_constructor_base):
    class _PotentialAtElectrodeMasked(
            ckESI_kernel_constructor_base._PotentialAtElectrodeMasked):
        @property
        def POT_XYZ(self):
            return np.meshgrid(self.convolver.POT_X,
                               self.convolver.POT_Y,
                               self.convolver.POT_Z,
                               indexing='ij')

    class _PotentialAtElectrodeAnalyticalMasked(
            _PotentialAtElectrodeMasked,
            ckESI_kernel_constructor_base._PotentialAtElectrodeLeadfieldForbiddenMask,
            ckESI_kernel_constructor_base._PotentialAtElectrodeAnalytical):
        pass

    class _PotentialAtElectrodeNumerical(
            ckESI_kernel_constructor_base._PotentialAtElectrodeNumerical,
            ckESI_kernel_constructor_base._PotentialAtElectrodePotAttribute):
        pass

    class _PotentialAtElectrodeNumericalMasked(ckESI_kernel_constructor_base._PotentialAtElectrodeNumericalMask,
                                               _PotentialAtElectrodeMasked):
        pass


class ckCSD_kernel_constructor_MOI(ckESI_kernel_constructor):
    def __init__(self,
                 model_source,
                 convolver,
                 source_indices,
                 csd_indices,
                 electrodes,
                 slice_thickness,
                 slice_conductivity,
                 saline_conductivity,
                 glass_conductivity=0.0,
                 n=128
                 ):
        self.slice_thickness = slice_thickness
        self.slice_conductivity = slice_conductivity
        self.saline_conductivity = saline_conductivity
        self.glass_conductivity = glass_conductivity
        self.n = n
        super(ckCSD_kernel_constructor_MOI,
              self).__init__(model_source,
                             convolver,
                             source_indices,
                             csd_indices,
                             electrodes)

    def _create_pre_kernel(self, electrodes, weights):
        wtg = float(self.slice_conductivity - self.glass_conductivity) / (
                    self.slice_conductivity + self.glass_conductivity)
        wts = float(self.slice_conductivity - self.saline_conductivity) / (
                    self.slice_conductivity + self.saline_conductivity)

        weights = [1.0]
        for i in range(self.n):
            weights.append(wtg ** i * wts ** (i + 1))
            weights.append(wtg ** (i + 1) * wts ** i)

        for i in range(1, self.n + 1):
            weights.append((wtg * wts) ** i)
            weights.append((wtg * wts) ** i)

        weights = np.array(weights)

        SRC_X, SRC_Y, SRC_Z = np.meshgrid(self.convolver.SRC_X,
                                          self.convolver.SRC_Y,
                                          self.convolver.SRC_Z,
                                          indexing='ij')
        SRC_X = SRC_X[self.source_indices].reshape(-1, 1)
        SRC_Y = SRC_Y[self.source_indices].reshape(-1, 1)
        SRC_Z = SRC_Z[self.source_indices].reshape(-1, 1)
        n_bases = SRC_X.size

        self._pre_kernel = np.full((n_bases, len(electrodes)),
                                   np.nan)
        for i_ele, electrode in enumerate(electrodes):
            ele_z = [electrode.z]
            for i in range(self.n):
                ele_z.append(2 * (i + 1) * self.slice_thickness - electrode.z)
                ele_z.append(-2 * i * self.slice_thickness - electrode.z)

            for i in range(1, self.n + 1):
                ele_z.append(electrode.z + 2 * i * self.slice_thickness)
                ele_z.append(electrode.z - 2 * i * self.slice_thickness)

            ele_z = np.reshape(ele_z, (1, -1))

            POT = self.model_source.potential(electrode.x - SRC_X,
                                              electrode.y - SRC_Y,
                                              ele_z - SRC_Z)

            self._pre_kernel[:, i_ele] = np.matmul(POT, weights)
        self._pre_kernel /= n_bases


if __name__ == '__main__':
    import itertools

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
