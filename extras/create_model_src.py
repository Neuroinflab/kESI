#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2023 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
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

from kesi.common import SphericalSplineSourceBase


parser = argparse.ArgumentParser(description="Create a sigmoid model source definition.")
parser.add_argument("-d", "--definition",
                    nargs='+',
                    required=True,
                    metavar="<definition.json>",
                    help="output file(s)")
parser.add_argument("-r", "--radius",
                    nargs='+',
                    required=True,
                    type=float,
                    metavar="<source radius [m]>",
                    help="radius (radii) of the source(s); cycles if not enough values given")

args = parser.parse_args()

for file, src_r in zip(args.definition, cycle(args.radius)):
    sd = src_r / 3.0
    nodes = [sd, src_r]
    coefficients = [[1],
                    [0,
                     2.25 / sd,
                     -1.5 / sd ** 2,
                     0.25 / sd ** 3]]
    src = SphericalSplineSourceBase(0, 0, 0, nodes, coefficients)
    k = src.csd(0, 0, 0)
    k_coefficients = [[k * v for v in vs] for vs in coefficients]
    normalized_src = SphericalSplineSourceBase(0.0, 0.0, 0.0,
                                               nodes,
                                               k_coefficients)

    with open(file, "w") as fh:
        normalized_src.toJSON(fh)
