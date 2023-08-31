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

from kesi import Reconstructor

class RegularizationParameter(object):
    def __init__(self, CV : pd.DataFrame = None, default : float = 0.0):
        self.CV = CV
        self.default = default

    def __call__(self, name : str) -> float:
        if self.CV is None:
            return self.default

        return  self.CV.REGULARIZATION_PARAMETER[self.CV[name].argmin()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model CSD in sphere on plate geometry with FEM.")
    parser.add_argument("-o", "--output",
                        metavar="<CSDs.npz>",
                        dest="output",
                        required=True,
                        help="path to the CSDs file")
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
    parser.add_argument("-c", "--crosskernel",
                        metavar="<crosskernel.npz>",
                        dest="crosskernel",
                        required=True,
                        help="path to the crosskernel file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--reguralization-parameter",
                       metavar="<regularization parameter>",
                       type=float,
                       dest="regularization_parameter",
                       default=0,
                       help="regularization parameter")
    group.add_argument("--cross-validation",
                       metavar="<errors.csv>",
                       dest="cross_validation",
                       help="path to the CV errors file")
    args = parser.parse_args()


    POTENTIALS = pd.read_csv(args.potentials)
    with np.load(args.kernel) as fh:
        KERNEL = fh["KERNEL"]

    with np.load(args.crosskernel) as fh:
        CROSSKERNEL = fh["CROSSKERNEL"]
        output = {k: fh[k] for k in "XYZ" if k in fh}

    reconstructor = Reconstructor(KERNEL, CROSSKERNEL)

    regularization_parameter = RegularizationParameter(
        pd.read_csv(args.cross_validation)
        if args.cross_validation is not None
        else None,
        args.regularization_parameter)

    names = [name for name in POTENTIALS.columns if name.startswith("POTENTIAL")]
    CSD = np.full(CROSSKERNEL.shape[:-1] + (len(names),),
                  np.nan)
    output["CSD"] = CSD

    _idx = (slice(0, None),) * (len(CROSSKERNEL.shape) - 1)
    for i, name in enumerate(names):
        CSD[_idx + (i,)] = reconstructor(POTENTIALS[name].to_numpy().copy(),
                                         regularization_parameter(name))

    np.savez_compressed(args.output, **output)
