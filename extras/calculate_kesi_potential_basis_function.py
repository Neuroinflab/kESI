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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy.integrate as si

import _fast_reciprocal_reconstructor as frr
import _common_new as common
from electrodes import ElectrodeIntegrationNodesAtSamplingGrid as Electrode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate values of kCSD potential basis functions at electrode.")
    parser.add_argument("-o", "--output",
                        metavar="<output>",
                        dest="output",
                        help="output directory")
    parser.add_argument("-i", "--input",
                        metavar="<input>",
                        dest="input",
                        help="input directory")
    parser.add_argument("-c", "--centroids",
                        required=True,
                        metavar="<centroids.npz>",
                        help="centroids grid with mask")
    parser.add_argument("-s", "--source",
                        required=True,
                        metavar="<source.json>",
                        help="definition of shape of CSD basis function")
    parser.add_argument("names",
                        metavar="<electrode name>",
                        nargs="+",
                        help="names of electrodes")

    args = parser.parse_args()


    with np.load(args.centroids) as fh:
        CENTROID_XYZ = [fh[c] for c in ["X", "Y", "Z"]]
        CENTROID_MASK = fh["MASK"]

    for name in args.names:
        model_src = common.SphericalSplineSourceKCSD.fromJSON(open(args.source))
        electrode = Electrode(os.path.join(args.input, f"{name}.npz"))

        d_xyz = np.array([(A[-1] - A[0]) / (len(A) - 1)
                          for A in electrode.SAMPLING_GRID])
        _ns = np.ceil(model_src.radius / d_xyz)
        romberg_ks = 1 + np.ceil(np.log2(_ns)).astype(int)
        romberg_weights = tuple(si.romb(np.identity(2 ** k + 1)) * 2.0 ** -k
                                for k in romberg_ks)

        LF_XYZ = [A[(A >= C.min() - model_src.radius)
                    & (A <= C.max() + model_src.radius)]
                  for A, C in zip(electrode.SAMPLING_GRID,
                                  CENTROID_XYZ)]

        convolver = frr.Convolver(LF_XYZ, LF_XYZ)

        CENTROIDS_IN_SRC = [np.isin(C.flatten(), S)
                            for S, C in zip(convolver.SRC_GRID, CENTROID_XYZ)]

        for c, IDX in zip("XYZ", CENTROIDS_IN_SRC):
            if not IDX.all():
                logger.warning(f"{(~IDX).sum()} centroid grid nodes missing along the {c} axis")

        SRC_IN_CENTROIDS = [np.isin(S.flatten(), C)
                            for S, C in zip(convolver.SRC_GRID, CENTROID_XYZ)]

        SRC_MASK = np.zeros(convolver.shape("SRC"),
                            dtype=bool)
        SRC_MASK[np.ix_(*SRC_IN_CENTROIDS)] = CENTROID_MASK[
                                                      np.ix_(*CENTROIDS_IN_SRC)]

        convolver_interface = frr.ConvolverInterfaceIndexed(convolver,
                                                            model_src.csd,
                                                            romberg_weights,
                                                            SRC_MASK)

        pbf = frr.pbf.AnalyticalCorrectedNumerically(
                                                  convolver_interface,
                                                  potential=model_src.potential)

        with pbf:
            np.savez_compressed(os.path.join(args.output,
                                             f"{name}.npz"),
                                POTENTIALS=pbf(electrode),
                                X=electrode.x,
                                Y=electrode.y,
                                Z=electrode.z)
