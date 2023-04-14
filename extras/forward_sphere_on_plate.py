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

import numpy as np
import pandas as pd
import scipy.interpolate as si

from forward_model import Spherical as ForwardModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model CSD in sphere on plate geometry with FEM.")
    parser.add_argument("-o", "--output",
                        metavar="<images.csv>",
                        dest="output",
                        required=True,
                        help="path to the images file")
    parser.add_argument("-s", "--sources",
                        metavar="<sources.npz>",
                        dest="sources",
                        required=True,
                        help="path to the file containing sources (CSD)")
    parser.add_argument("-e", "--electrodes",
                        metavar="<electrodes.csv>",
                        dest="electrodes",
                        required=True,
                        help="path to the electrode location config file")
    parser.add_argument("-c", "--config",
                        metavar="<config.ini>",
                        dest="config",
                        required=True,
                        help="path to the model config file")
    parser.add_argument("-m", "--mesh",
                        metavar="<mesh.xdmf>",
                        dest="mesh",
                        required=True,
                        help="path to the FEM mesh")
    parser.add_argument("-d", "--degree",
                        type=int,
                        metavar="<FEM element degree>",
                        dest="degree",
                        help="degree of FEM elements",
                        default=1)
    parser.add_argument("--element-type",
                        metavar="<FEM element type>",
                        dest="element_type",
                        help="type of FEM elements",
                        default="CG")
    parser.add_argument("-g", "--grounded-plate-edge-z",
                        type=float,
                        dest="grounded_plate_edge_z",
                        metavar="<grounded plate edge's z>",
                        help="Z coordinate of the grounded plate",
                        default=-0.088)
    parser.add_argument("-q", "--quiet",
                        dest="quiet",
                        action="store_true",
                        help="do not print results",
                        default=False)
    parser.add_argument("--start-from",
                        type=int,
                        metavar="<source number>",
                        dest="start_from",
                        help="number of the first source to start from (useful in case of broken run)",
                        default=0)

    args = parser.parse_args()


    DF = pd.read_csv(args.electrodes)
    ELECTRODE_LOCATION = list(zip(DF.X, DF.Y, DF.Z))

    fem = ForwardModel(args.mesh, args.degree, args.config,
                       grounded_plate_at=args.grounded_plate_edge_z,
                       element_type=args.element_type,
                       quiet=args.quiet)

    with np.load(args.sources) as fh:
        XYZ = [fh[x].flatten() for x in "XYZ"]
        CSD = fh["CSD"]
        for i in range(args.start_from, CSD.shape[-1]):
            csd = si.RegularGridInterpolator(XYZ, CSD[:, :, :, i],
                                             bounds_error=False,
                                             fill_value=0)
            potential = fem(csd)
            DF[f"SOURCE_{i}"] = [potential(*xyz) for xyz in ELECTRODE_LOCATION]
            DF.to_csv(args.output,
                      index=False)
