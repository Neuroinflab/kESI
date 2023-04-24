#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2023 Jakub M. Dzik (Institute of Applied Psychology;       #
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
from itertools import cycle

import numpy as np

from common import shape

parser = argparse.ArgumentParser(description="Create sampling grid.")
parser.add_argument("-g", "--grid",
                    required=True,
                    metavar="<grid.npz>",
                    help="output file")
parser.add_argument("-c", "--coords",
                    default="XYZ",
                    metavar="<coordinate system>",
                    help="a string containing one-letter label of grid coords (like the default 'XYZ')")
parser.add_argument("-s", "--start",
                    nargs="+",
                    default=[0.0],
                    type=float,
                    dest="starts",
                    metavar="<start>",
                    help="beginning of the grid along each axis (cycles if not enough values given)")
parser.add_argument("-e", "--end",
                    nargs="+",
                    required=True,
                    type=float,
                    dest="ends",
                    metavar="<end>",
                    help="ends of the grid along each axis (cycles if not enough values given)")
group = parser.add_mutually_exclusive_group()
group.add_argument("-n",
                   nargs="+",
                   type=int,
                   dest="ns",
                   metavar="<n>",
                   help="number of grid nodes along each axis (cycles if not enough values given)")
group.add_argument("-k",
                   nargs="+",
                   default=[0],
                   type=int,
                   dest="ks",
                   metavar="<k>",
                   help="k exponent of number of grid nodes along each axis defined as 2**k + 1 (cycles if not enough values given)")
args = parser.parse_args()

ns = args.ns if args.ns is not None else [2**k + 1 for k in args.ks]
dimensionality = len(args.coords)

np.savez_compressed(
    args.grid,
    **{coord: np.linspace(start, end, n_nodes).reshape(shape(dimensionality,
                                                             dimension))
       for dimension, (coord, n_nodes, start, end)
       in enumerate(zip(args.coords,
                        cycle(ns),
                        cycle(args.starts),
                        cycle(args.ends)))

       })
