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
from kesi.kernel.electrode import IntegrationNodesAtSamplingGrid as Electrode

from _calculate_potential_basis_function import ScriptBase


class Script(ScriptBase):
    PotentialBasisFunction = pbf.AnalyticalCorrectedNumerically

    class ArgumentParser(ScriptBase.ArgumentParser):
        def __init__(self):
            super().__init__("kESI")
            self.add_argument("-i", "--input",
                              metavar="<input>",
                              dest="input",
                              help="input directory")

    def Electrode(self, directory):
        class _Electrode(Electrode):
            def __init__(self, name):
                super().__init__(os.path.join(directory, f"{name}.npz"))
                self.name = name

        return _Electrode

    def _load_electrodes(self, args):
        self._electrodes = collections.defaultdict(dict)
        for electrode in map(self.Electrode(args.input), args.names):
            self._store_electrode(electrode)

    def _store_electrode(self, electrode):
        key = tuple(map(tuple, electrode.SAMPLING_GRID))
        self._electrodes[key][electrode.name] = electrode

    def run(self):
        for sampling_grid, electrodes in self._electrodes.items():
            self._run(electrodes, sampling_grid)

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
