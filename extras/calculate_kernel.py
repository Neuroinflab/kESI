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
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create kernel.")
    parser.add_argument("-k", "--kernel",
                        metavar="<kernel.npz>",
                        dest="kernel",
                        help="kernel matrix")
    parser.add_argument("-p", "--pbf",
                        metavar="<pbf.npz>",
                        dest="pbf",
                        help="values of potential basis functions at electrodes")
    parser.add_argument("-a", "--analysis",
                        metavar="<analysis.npz>",
                        dest="analysis",
                        help="auxiliary analytical data")
    parser.add_argument("-i", "--input",
                        required=True,
                        metavar="<input>",
                        dest="input",
                        help="input directory")
    parser.add_argument("-e", "--electrodes",
                        required=True,
                        metavar="<electrodes.csv>",
                        help="locations of electrodes")

    args = parser.parse_args()

    ELECTRODES = pd.read_csv(args.electrodes,
                             index_col="NAME",
                             usecols=["NAME", "X", "Y", "Z"])

    for i, name in enumerate(ELECTRODES.index):
        with np.load(os.path.join(args.input, f"{name}.npz")) as fh:
            COL = fh["POTENTIALS"]
            if i == 0:
                B = np.full((len(COL), len(ELECTRODES)), np.nan)

            B[:, i] = COL

    if args.pbf is not None:
        np.savez_compressed(args.pbf, B=B)

    if args.kernel is not None:
        np.savez_compressed(args.kernel, KERNEL=np.matmul(B.T, B))

    if args.analysis is not None:
        _U, _S, _V = np.linalg.svd(B,
                                   full_matrices=False,
                                   compute_uv=True)
        del B

        np.savez_compressed(args.analysis,
                            EIGENVALUES=np.square(_S),
                            EIGENSOURCES=_U,
                            SINGULARVALUES=_S,
                            EIGENVECTORS=_V.T)
