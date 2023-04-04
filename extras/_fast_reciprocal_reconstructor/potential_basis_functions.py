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


def _sum_of_not_none(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        V, V_SUPER = f(self, *args, **kwargs)
        if V_SUPER is not None:
            V += V_SUPER
        return V
    return wrapper


class _Base(object):
    def __init__(self, convolver_interface):
        self.convolver_interface = convolver_interface

    def __call__(self, electrode):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _PotAttribute(_Base):
    def __enter__(self):
        self.POT_XYZ = self.convolver_interface.meshgrid('POT')
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.POT_XYZ
        super().__exit__(exc_type, exc_val, exc_tb)


class _PotProperty(object):
    @property
    def POT_XYZ(self):
        return self.convolver_interface.meshgrid('POT')


class _FromLeadfield(_Base):
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


class _MaskedLeadfield(_FromLeadfield):
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


class _LeadfieldCroppingAnalyticalBasesNumerically(_MaskedLeadfield):
    """
    `.POT_XYZ` attribute/property required
    """
    def __enter__(self):
        cm = super().__enter__()
        self.csd_forbidden_mask = ~self.leadfield_allowed_mask
        self.POT_XYZ_CROPPED = [A[self.csd_forbidden_mask] for A in self.POT_XYZ]
        return cm

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


class _PotMaskedAttribute(_MaskedLeadfield,
                          _PotProperty):
    """
    `.POT_XYZ` attribute/property required
    """
    def __enter__(self):
        cm = super().__enter__()
        self.POT_XYZ_MASKED = [A[self.leadfield_allowed_mask] for A in self.POT_XYZ]
        return cm

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.POT_XYZ_MASKED
        super().__exit__(exc_type, exc_val, exc_tb)


class NumericalMasked(_PotMaskedAttribute):
    @_sum_of_not_none
    def _allowed_leadfield(self, electrode):
        return (electrode.leadfield(*self.POT_XYZ_MASKED),
                super()._allowed_leadfield(electrode))


class _FromLeadfieldNotMasked(_FromLeadfield):
    def _create_leadfield(self, electrode):
        self.LEADFIELD = None


class _LeadfieldFromElectrode(_FromLeadfieldNotMasked):
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


class Analytical(_Base):
    def __init__(self, convolver_interface, potential, **kwargs):
        super().__init__(convolver_interface, **kwargs)
        self.potential = potential

    def __enter__(self):
        self.SRC_X, self.SRC_Y, self.SRC_Z = self.convolver_interface.src_coords()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.SRC_X, self.SRC_Y, self.SRC_Z
        super().__exit__(exc_type, exc_val, exc_tb)

    @_sum_of_not_none
    def __call__(self, electrode):
        return (self._potential_divided_by_relative_conductivity_if_available(
                                                                     electrode),
                super().__call__(electrode))

    def _potential_divided_by_relative_conductivity_if_available(self,
                                                                 electrode):
        return self._divide_by_relative_conductivity_if_available(
                                       self.potential(electrode.x - self.SRC_X,
                                                      electrode.y - self.SRC_Y,
                                                      electrode.z - self.SRC_Z),
                                       electrode)

    def _divide_by_relative_conductivity_if_available(self,
                                                      potential,
                                                      electrode):
        try:
            factor = 1.0 / electrode.conductivity

        except AttributeError:
            return potential

        return factor * potential


class Numerical(_LeadfieldFromElectrode,
                _PotAttribute):
    pass


class AnalyticalMaskedNumerically(_PotProperty,
                                  _LeadfieldCroppingAnalyticalBasesNumerically,
                                  Analytical):
    _LEADFIELD_METHOD = 'leadfield'


class _LeadfieldFromMaskedCorrectionPotential(_PotMaskedAttribute):
    @_sum_of_not_none
    def _allowed_leadfield(self, electrode):
        return (electrode.correction_leadfield(*self.POT_XYZ_MASKED),
                super()._allowed_leadfield(electrode))


class AnalyticalMaskedAndCorrectedNumerically(
                                   _LeadfieldCroppingAnalyticalBasesNumerically,
                                   _LeadfieldFromMaskedCorrectionPotential,
                                   Analytical):
    _LEADFIELD_METHOD = 'base_leadfield'

    def _provide_leadfield_array(self):
        self.alloc_leadfield_if_necessary()

    def alloc_leadfield_if_necessary(self):
        if not hasattr(self, 'LEADFIELD'):
            self.LEADFIELD = self.convolver_interface.empty('POT')


class NumericalCorrection(_FromLeadfieldNotMasked,
                          _PotAttribute):
    def _create_leadfield(self, electrode):
        super()._create_leadfield(electrode)
        LEADFIELD = electrode.correction_leadfield(*self.POT_XYZ)
        if self.LEADFIELD is not None:
            self.LEADFIELD += LEADFIELD
        else:
            self.LEADFIELD = LEADFIELD


class AnalyticalCorrectedNumerically(NumericalCorrection,
                                     Analytical):
    pass
