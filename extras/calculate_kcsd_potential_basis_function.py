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
import collections

import numpy as np
import pandas as pd

import _fast_reciprocal_reconstructor as frr
import _common_new as common


Electrode = collections.namedtuple("Electrode", ['x', 'y', 'z'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate values of kCSD potential basis functions at electrode.")
    parser.add_argument("-o", "--output",
                        metavar="<output>",
                        dest="output",
                        help="output directory")
    parser.add_argument("-c", "--centroids",
                        required=True,
                        metavar="<centroids.npz>",
                        help="centroids grid with mask")
    parser.add_argument("-s", "--source",
                        required=True,
                        metavar="<source.json>",
                        help="definition of shape of CSD basis function")
    parser.add_argument("-e", "--electrodes",
                        required=True,
                        metavar="<electrodes.csv>",
                        help="locations of electrodes")
    parser.add_argument("--conductivity",
                        type=float,
                        default=0.33,
                        metavar="<conductivity [S/m]>",
                        help="medium conductivity")
    parser.add_argument("names",
                        metavar="<electrode name>",
                        nargs="+",
                        help="names of electrodes")

    args = parser.parse_args()

    ELECTRODES = pd.read_csv(args.electrodes,
                             index_col="NAME",
                             usecols=["NAME", "X", "Y", "Z"]).loc[args.names]

    with np.load(args.centroids) as fh:
        X, Y, Z, MASK = [fh[c] for c in ["X", "Y", "Z", "MASK"]]

    convolver = frr.Convolver([X, Y, Z],
                              [X, Y, Z])

    model_src = common.SphericalSplineSourceKCSD.fromJSON(
                                                open(args.source),
                                                conductivity=args.conductivity)

    convolver_interface = frr.ConvolverInterfaceIndexed(convolver,
                                                        model_src.csd,
                                                        [],
                                                        MASK)

    pae = frr.PAE_Analytical(convolver_interface,
                             potential=model_src.potential)

    with pae:
        for name, loc in ELECTRODES.iterrows():
            electrode = Electrode(*loc)

            np.savez_compressed(os.path.join(args.output,
                                             f"{name}.npz"),
                                POTENTIALS=(pae(electrode)),
                                CONDUCTIVITY=args.conductivity,
                                X=electrode.x,
                                Y=electrode.y,
                                Z=electrode.z)
