#!/usr/bin/env python
# coding: utf-8
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
import datetime

import numpy as np

import local.fem.common as fc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample slice FEM solutions.")
    parser.add_argument("-o", "--output",
                        dest="output",
                        required=True,
                        metavar="<output.npz>",
                        help="path to the output file")
    parser.add_argument("-c", "--config",
                        dest="config",
                        required=True,
                        metavar="<config.ini>",
                        help="path to the solution metadata")
    parser.add_argument("-g", "--grid",
                        dest="grid",
                        required=True,
                        metavar="<sampling_grid.npz>",
                        help="path to the sampling grid definition")
    parser.add_argument("-f", "--fill",
                        dest="fill",
                        type=float,
                        default=np.nan,
                        metavar="<fill value>",
                        help="fill value")
    parser.add_argument("-q", "--quiet",
                        default=False,
                        dest="quiet",
                        action="store_true",
                        help="do not print results")

    args = parser.parse_args()

    with np.load(args.grid) as fh:
        X, Y, Z = [fh[c] for c in "XYZ"]

    config = fc.MetadataReader(args.config)

    preprocess_start = datetime.datetime.now()
    function_manager = fc.FunctionManager(config.getpath("fem", "mesh"),
                                          config.getint("fem", "degree"),
                                          config.get("fem", "element_type"))
    loading_start = datetime.datetime.now()

    correction_potential = function_manager.load(config.getpath("correction",
                                                                "filename"))

    ERROR_R = []
    start = datetime.datetime.now()

    CORRECTION_POTENTIAL = np.full((X.shape[0],
                                    Y.shape[1],
                                    Z.shape[2]),
                                   args.fill)

    if not args.quiet:
        print(f"PREPROCESSING: {loading_start - preprocess_start}")
        print(f"LOADING: {start - loading_start}")

    XF, YF, ZF = [A.flatten() for A in [X, Y, Z]]

    for i_x, x in enumerate(XF):
        if not args.quiet:
            print(
                f"{i_x}\t{x * 1000_000:.1f}um\t{datetime.datetime.now() - start}")

        for i_y, y in enumerate(YF):
            for i_z, z in enumerate(ZF):
                try:
                    CORRECTION_POTENTIAL[i_x, i_y, i_z] = correction_potential(x, y, z)
                except RuntimeError:
                    ERROR_R.append((x, y, z))

    np.savez_compressed(args.output,
                        CORRECTION_POTENTIAL=CORRECTION_POTENTIAL,
                        X=X,
                        Y=Y,
                        Z=Z,
                        LOCATION=[config.getfloat("electrode", c) for c in
                                  "xyz"],
                        BASE_CONDUCTIVITY=config.getfloat("correction",
                                                          "base_conductivity"),
                        _PREPROCESSING_TIME=(
                                loading_start - preprocess_start).total_seconds(),
                        _LOADING_TIME=(
                                start - loading_start).total_seconds(),
                        _PROCESSING_TIME=(
                                datetime.datetime.now() - start).total_seconds())

    if ERROR_R != []:
        print("ERROR R:")
        print(ERROR_R)