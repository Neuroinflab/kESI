#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2021 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

import _fast_reciprocal_reconstructor as frr
import common


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create volumetric eigensources.")
    parser.add_argument("-o", "--output",
                        required=True,
                        metavar="<output.npz>",
                        dest="output",
                        help="volumetric eigensources")
    parser.add_argument("-i", "--input",
                        required=True,
                        metavar="<analysis.npz>",
                        dest="input",
                        help="auxiliary kernel analysis matrices")
    parser.add_argument("-c", "--centroids",
                        required=True,
                        metavar="<centroids.npz>",
                        help="centroids grid with mask")
    parser.add_argument("-g", "--grid",
                        required=True,
                        metavar="<grid.npz>",
                        help="CSD grid")
    parser.add_argument("-s", "--source",
                        required=True,
                        metavar="<source.json>",
                        help="definition of shape of CSD basis function")
    parser.add_argument("--coords",
                        default="XYZ",
                        metavar="<coordinate system>",
                        help="a string containing one-letter label of grid coords (like the default 'XYZ')")

    args = parser.parse_args()

    with np.load(args.centroids) as fh:
        centroids = [fh[c] for c in args.coords]
        SRC_MASK = fh["MASK"]

    with np.load(args.grid) as fh:
        csd_grid = [fh[c] for c in args.coords]

    model_src = common.SphericalSplineSourceBase.fromJSON(open(args.source))

    convolver = frr.Convolver(centroids, csd_grid)

    src_diameters = [int(2 * np.floor(model_src.radius / _d)) + 1
                     for _d in convolver.steps("CSD")]

    with np.load(args.input) as fh:
        CANONICAL_ES = fh["EIGENSOURCES"]

    EIGENSOURCES = np.full(convolver.shape("CSD") + CANONICAL_ES.shape[1:],
                           np.nan)
    _SRC = np.zeros(convolver.shape("SRC"))
    for i, _SRC[SRC_MASK] in enumerate(CANONICAL_ES.T):
        EIGENSOURCES[:, :, :, i] = convolver.basis_functions_weights_to_csd(_SRC,
                                                                            model_src.csd,
                                                                            src_diameters)
    del CANONICAL_ES, _SRC

    np.savez_compressed(os.path.join(args.output),
                        CSD=EIGENSOURCES,
                        X=convolver.CSD_X,
                        Y=convolver.CSD_Y,
                        Z=convolver.CSD_Z)
