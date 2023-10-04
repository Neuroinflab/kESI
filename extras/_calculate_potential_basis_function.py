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

import argparse
import os

import numpy as np

from kesi.common import SphericalSplineSourceKCSD
from kesi.kernel.constructor import Convolver, ConvolverInterfaceIndexed


class ScriptBase(object):
    class ArgumentParser(argparse.ArgumentParser):
        def __init__(self, method):
            super().__init__(
                description=f"Calculate values of {method} potential basis functions at electrode.")
            self.add_argument("names",
                              metavar="<electrode name>",
                              nargs="+",
                              help="names of electrodes")
            self.add_argument("-o", "--output",
                              metavar="<output>",
                              dest="output",
                              help="output directory")
            self.add_argument("-c", "--centroids",
                              required=True,
                              metavar="<centroids.npz>",
                              help="centroids grid with mask")
            self.add_argument("-s", "--source",
                              required=True,
                              metavar="<source.json>",
                              help="definition of shape of CSD basis function")

    def __init__(self):
        self._init(self.parse_args())

    def parse_args(self):
        parser = self.ArgumentParser()
        return parser.parse_args()

    def _init(self, args):
        self._load_electrodes(args)
        self._output_directory = args.output
        self._load_centroids(args.centroids)
        self._model_src = SphericalSplineSourceKCSD.fromJSON(open(args.source))

    def _load_centroids(self, file):
        with np.load(file) as fh:
            self.CENTROID_XYZ = [fh[c] for c in ["X", "Y", "Z"]]
            self.CENTROID_MASK = fh["MASK"]

    def _run(self, electrodes, *args, **kwargs):
        self._store_values_at_electrodes(self._get_potential_basis_function(*args, **kwargs),
                                         electrodes)

    def _get_potential_basis_function(self, *args, **kwargs):
        weights = self._get_quadrature(*args, **kwargs)
        convolver = self._get_convolver(*args, **kwargs)
        convolver_interface = self._get_convolver_interface(convolver, weights)
        return self.PotentialBasisFunction(convolver_interface,
                                           potential=self._model_src.potential)

    def _get_convolver(self, *args, **kwargs):
        convolver_grid = self._get_convolver_grid(*args, **kwargs)
        return Convolver(convolver_grid, convolver_grid)

    def _get_convolver_interface(self, convolver, quadrature):
        SRC_MASK = self._get_src_mask(convolver)
        return ConvolverInterfaceIndexed(convolver,
                                         self._model_src.csd,
                                         quadrature,
                                         SRC_MASK)

    def _store_values_at_electrodes(self, potential_basis_function, electrodes):
        with potential_basis_function:
            for name, electrode in electrodes.items():
                self._save_potential_basis_function(f"{name}.npz",
                                                    potential_basis_function,
                                                    electrode)

    def _save_potential_basis_function(self,
                                       filename,
                                       potential_basis_function,
                                       electrode):
        np.savez_compressed(self._filepath(filename),
                            POTENTIALS=potential_basis_function(electrode),
                            CONDUCTIVITY=electrode.conductivity,
                            X=electrode.x,
                            Y=electrode.y,
                            Z=electrode.z)

    def _filepath(self, filename):
        return os.path.join(self._output_directory, filename)

    @property
    def src_radius(self):
        return self._model_src.radius
