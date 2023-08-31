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

from kesi.common import cv
from kesi._engine import _LinearKernelSolver as Solver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate leave-one-out cross-validation errors.")
    parser.add_argument("-o", "--output",
                        metavar="<errors.csv>",
                        dest="output",
                        required=True,
                        help="path to the CV errors file")
    parser.add_argument("-p", "--potentials",
                        metavar="<potentials.csv>",
                        dest="potentials",
                        required=True,
                        help="path to the file containing potentials")
    parser.add_argument("-k", "--kernel",
                        metavar="<kernel.npz>",
                        dest="kernel",
                        required=True,
                        help="path to the kernel file")
    group = parser.add_argument_group("Tested regularization parameters",
                 """Defines sequence of tested parameters spaced evenly
                    on a log scale.  Semantics of arguments is similar
                    to arguments of `numpy.logspace()`""")
    group.add_argument("-s", "--start",
                       metavar="<start>",
                       dest="start",
                       required=True,
                       type=float,
                       default=-5,
                       help="lg10 of the starting value of the sequence")
    group.add_argument("-e", "--end", "--stop",
                       metavar="<stop>",
                       dest="stop",
                       required=True,
                       default=15,
                       type=float,
                       help="lg10 of the final value of the sequence")
    group.add_argument("-n", "--num",
                       metavar="<start>",
                       dest="n",
                       type=int,
                       default=20 * 3 + 1,
                       help="length of the sequence")

    args = parser.parse_args()

    REGULARIZATION_PARAMETERS = np.logspace(args.start, args.stop, args.n)
    with np.load(args.kernel) as fh:
        solver = Solver(fh["KERNEL"])

    POTENTIALS = pd.read_csv(args.potentials)
    DF = pd.DataFrame({"REGULARIZATION_PARAMETER": REGULARIZATION_PARAMETERS})

    for name in POTENTIALS.columns:
        if name.startswith("POTENTIAL_"):
            DF[name] = cv(solver,
                          POTENTIALS[name].to_numpy(copy=True),
                          REGULARIZATION_PARAMETERS)

    DF.to_csv(args.output,
              index=False)
