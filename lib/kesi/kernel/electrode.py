#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Institute of Applied Psychology;  #
#    Faculty of Management and Social Communication; Jagiellonian University) #
#    Copyright (C) 2023 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
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

import numpy as np
import scipy.interpolate as si


class _Base(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Conductivity(_Base):
    def __init__(self, x, y, z, conductivity, **kwargs):
        super().__init__(x=x, y=y, z=z, **kwargs)
        self.conductivity = conductivity


class _LeadfieldCorrectionBase(Conductivity):
    def __init__(self, filename, **kwargs):
        """
        Parameters
        ----------

        filename : str
            Path to the sampled correction potential.
        """
        self.filename = filename
        with np.load(filename) as fh:
            self._call_super_init_with_read_arguments(fh, kwargs)
            self.SAMPLING_GRID = [fh[c].flatten() for c in "XYZ"]

    def _call_super_init_with_read_arguments(self, fh, kwargs):
        kwargs.update(zip("xyz", fh["LOCATION"]))
        super().__init__(conductivity=fh["BASE_CONDUCTIVITY"],
                         **kwargs)

    def correction_leadfield(self, X, Y, Z):
        """
        Correction of the leadfield of the electrode
        for violation of kCSD assumptions

        Parameters
        ----------
        X, Y, Z : np.array
            Coordinate matrices of the same shape.
        """
        with np.load(self.filename) as fh:
            return self._correction_leadfield(fh["CORRECTION_POTENTIAL"],
                                              [X, Y, Z])


class _Regularized(Conductivity):
    def __init__(self, dx=0, **kwargs):
        """
        Parameters
        ----------

        filename : str
            Path to the sampled correction potential.

        dx : float
            Integration step used to calculate a regularization
            parameter of the `.leadfield()` method.
        """
        super().__init__(**kwargs)
        self.dx = dx

    @property
    def _epsilon(self):
        """
        Regularization parameter of the `.leadfield()` method.

        Note
        ----

        The 0.15 factor choice has been based on a toy numerical experiment.
        Further, more rigorous experiments are definitely recommended.
        """
        return 0.15 * self.dx

    def _regularized_leadfield(self, X, Y, Z):
        """
        Regularized leadfield of the electrode in infinite homogenous
        isotropic medium (kCSD assumptions) of conductivity
        `.conductivity` S/m.

        Note
        ----

        The regularization is necessary to limit numerical integration
        errors.
        """
        return (0.25 / (np.pi * self.conductivity)
                / (self._epsilon
                   + np.sqrt(np.square(X - self.x)
                             + np.square(Y - self.y)
                             + np.square(Z - self.z))))

    # def leadfield(self, X, Y, Z):
    #     # For:
    #     #  - numerical kCSD,
    #     #  - numerical masking of analytical kCSD solutions.
    #     return self.base_leadfield(X, Y, Z)
    #
    # def leadfield(self, X, Y, Z):
    #     # For numerical kESI.
    #     return self.base_leadfield(X, Y, Z) + self.correction_leadfield(X, Y, Z)


class _InterpolatedLeadfieldCorrection(_LeadfieldCorrectionBase):
    def _correction_leadfield(self, SAMPLES, XYZ):
        return self._interpolate(SAMPLES, XYZ)

    def _interpolate(self, SAMPLES, XYZ):
        interpolator = si.RegularGridInterpolator(
            self.SAMPLING_GRID,
            SAMPLES,
            bounds_error=False,
            fill_value=0,
            method=self.interpolation_method)
        return interpolator(np.stack(XYZ, axis=-1))


class LinearlyInterpolatedLeadfieldCorrection(_InterpolatedLeadfieldCorrection):
    interpolation_method = "linear"


class NearestNeighbourInterpolatedLeadfieldCorrection(_InterpolatedLeadfieldCorrection):
    interpolation_method = "nearest"


class IntegrationNodesAtSamplingGrid(_LeadfieldCorrectionBase):
    def _correction_leadfield(self, SAMPLES, XYZ):
        # if XYZ points are in nodes of the sampling grid,
        # no time-consuming interpolation is necessary
        return SAMPLES[self._sampling_grid_indices(XYZ)]

    def _sampling_grid_indices(self, XYZ):
        return tuple(np.searchsorted(GRID, COORD)
                     for GRID, COORD in zip(self.SAMPLING_GRID, XYZ))
