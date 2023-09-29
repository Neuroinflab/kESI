#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
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

import argparse

import numpy as np
import pandas as pd


if __name__ == "__main__":
    def calculate_norms(X):
        BIAS = X.mean(axis=(0, 1, 2))
        X = abs(X)
        return {"BIAS": BIAS,
                "L1": X.mean(axis=(0, 1, 2)),
                "L2": np.sqrt(np.square(X).mean(axis=(0, 1, 2))),
                "LInf": X.max(axis=(0, 1, 2)),
                }

    parser = argparse.ArgumentParser(description="Calculate norms of CSD errors.")
    parser.add_argument("-o", "--output",
                        metavar="<errors.csv>",
                        dest="output",
                        required=True,
                        help="path to the CSD errors file")
    parser.add_argument("-c", "--csd",
                        metavar="<csd.npz>",
                        dest="csd",
                        required=True,
                        help="path to the file containing CSD profile")
    parser.add_argument("-r", "--reference",
                        metavar="<reference.npz>",
                        dest="reference",
                        required=True,
                        help="path to the file containing reference CSD profiles")

    args = parser.parse_args()

    with np.load(args.reference) as fh:
        REFERENCE = fh["CSD"]

    DF = pd.DataFrame({"SOURCE": range(REFERENCE.shape[-1])})
    for name, NORMS in calculate_norms(REFERENCE).items():
        DF[f"REFERENCE_{name}"] = NORMS

    with np.load(args.csd) as fh:
        for name, NORMS in calculate_norms(fh["CSD"] - REFERENCE).items():
            DF[f"ERROR_{name}"] = NORMS

    DF.to_csv(args.output,
              index=False)
