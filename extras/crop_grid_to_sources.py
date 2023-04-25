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

import numpy as np

from kesi import common


def support_bounds(centroids, r):
    return [(C.min() - r, C.max() + r) for C in centroids]

def cropped_grid(grid, bounds):
    return [G[(G >= low) & (G <= high)] for G, (low, high) in zip(grid, bounds)]

def shape_grid(grid):
    return [G.reshape(common.shape(len(grid), i)) for i, G in enumerate(grid)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce grid to support of CSD basis functions.")
    parser.add_argument("-i", "--input",
                        required=True,
                        metavar="<input.npz>",
                        dest="input",
                        help="grid to be cropped")
    parser.add_argument("-o", "--output",
                        required=True,
                        metavar="<output.npz>",
                        dest="output",
                        help="output grid")
    parser.add_argument("-c", "--centroids",
                        required=True,
                        metavar="<centroids.npz>",
                        help="centroids grid with mask")
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

    with np.load(args.input) as fh:
        src_grid = [fh[c] for c in args.coords]

    model_src = common.SphericalSplineSourceBase.fromJSON(open(args.source))
    dst_grid = cropped_grid(src_grid,
                            support_bounds(centroids,
                                           model_src.radius))
    np.savez_compressed(args.output,
                        **dict(zip(args.coords, shape_grid(dst_grid))))
