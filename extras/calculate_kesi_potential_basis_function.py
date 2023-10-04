#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

import os
import collections

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy.integrate as si

from kesi.kernel import potential_basis_functions as pbf
from kesi.kernel.electrode import (IntegrationNodesAtSamplingGrid,
                                   LinearlyInterpolatedLeadfieldCorrection)

from _calculate_potential_basis_function import ScriptBase


class Script(ScriptBase):
    PotentialBasisFunction = pbf.AnalyticalCorrectedNumerically

    class WrapElectrode(type):
        def __new__(cls, name, bases, attrs):
            try:
                class Electrode(attrs["Electrode"]):
                    def __init__(self, directory, name):
                        super().__init__(os.path.join(directory, f"{name}.npz"))
                        self.name = name

                attrs = attrs.copy()
                attrs["Electrode"] = Electrode
            except KeyError:
                pass
            return super().__new__(cls, name, bases, attrs)

    class _LeadfieldIntegratedBase(metaclass=WrapElectrode):
        def __init__(self, directory):
            self.directory = directory

        def get_electrode(self, name):
            return self.Electrode(self.directory, name)

    class LeadfieldIntegratedOnSaplingCorrectionGrid(_LeadfieldIntegratedBase):
        Electrode = IntegrationNodesAtSamplingGrid

        def get_sampling_grid(self, electrode):
            return tuple(map(tuple, electrode.SAMPLING_GRID))

    class LeadfieldIntegratedOnCustomGrid(_LeadfieldIntegratedBase):
        Electrode = LinearlyInterpolatedLeadfieldCorrection

        def __init__(self, directory, grid):
            super().__init__(directory)
            with np.load(grid) as fh:
                self.sampling_grid = tuple(tuple(fh[c].flatten())
                                           for c in "XYZ")

        def get_sampling_grid(self, electrode):
            return self.sampling_grid

    class ArgumentParser(ScriptBase.ArgumentParser):
        def __init__(self):
            super().__init__("kESI")
            self.add_argument("-i", "--input",
                              metavar="<input>",
                              dest="input",
                              help="input directory")
            self.add_argument("-g", "--grid",
                              metavar="<grid.npz>",
                              help="grid for integration of correction to the reciprocal potential of the electrode to be used instead of the sampling grid (integration is slower due to interpolation)")

    def _init(self, args):
        self._grid_handler = self._get_grid_handler(args)
        super()._init(args)

    def _get_grid_handler(self, args):
        if args.grid is not None:
            return self.LeadfieldIntegratedOnCustomGrid(args.input, args.grid)

        return self.LeadfieldIntegratedOnSaplingCorrectionGrid(args.input)

    def _load_electrodes(self, args):
        self._electrodes = collections.defaultdict(dict)

        for name in args.names:
            self._store_electrode(self._grid_handler.get_electrode(name))

    def _store_electrode(self, electrode):
        key = self._grid_handler.get_sampling_grid(electrode)
        self._electrodes[key][electrode.name] = electrode

    def run(self):
        for sampling_grid, electrodes in self._electrodes.items():
            self._run(electrodes, [np.array(A) for A in sampling_grid])

    def _get_convolver_grid(self, sampling_grid):
        return [A[(A >= C.min() - self.src_radius)
                  & (A <= C.max() + self.src_radius)]
                for A, C in zip(sampling_grid,
                                self.CENTROID_XYZ)]

    def _get_quadrature(self, sampling_grid):
        d_xyz = np.array([(A[-1] - A[0]) / (len(A) - 1)
                          for A in sampling_grid])
        _ns = np.ceil(self.src_radius / d_xyz)
        romberg_ks = 1 + np.ceil(np.log2(_ns)).astype(int)
        return tuple(si.romb(np.identity(2 ** k + 1)) * 2.0 ** -k
                     for k in romberg_ks)

    def _get_src_mask(self, convolver):
        CENTROIDS_IN_SRC = [np.isin(C.flatten(), S)
                            for S, C in zip(convolver.SRC_GRID,
                                            self.CENTROID_XYZ)]
        for c, IDX in zip("XYZ", CENTROIDS_IN_SRC):
            if not IDX.all():
                logger.warning(
                    f"{(~IDX).sum()} centroid grid nodes missing along the {c} axis")
        SRC_IN_CENTROIDS = [np.isin(S.flatten(), C)
                            for S, C in zip(convolver.SRC_GRID,
                                            self.CENTROID_XYZ)]
        SRC_MASK = np.zeros(convolver.shape("SRC"),
                            dtype=bool)
        SRC_MASK[np.ix_(*SRC_IN_CENTROIDS)] = self.CENTROID_MASK[
            np.ix_(*CENTROIDS_IN_SRC)]
        return SRC_MASK


if __name__ == "__main__":
    model = Script()
    model.run()
