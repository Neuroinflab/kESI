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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix appropriate eigensources.")
    parser.add_argument("-o", "--output",
                        required=True,
                        metavar="<mixture.npz>",
                        dest="output",
                        help="mixed eigensources")
    parser.add_argument("-a", "--analysis",
                        required=True,
                        nargs=2,
                        metavar="<analysis.npz>",
                        help="auxilary kernel analysis matrices")
    parser.add_argument("-e", "--eigensources",
                        required=True,
                        nargs=2,
                        metavar="<eigensources.npz>",
                        help="matching eigensource CSD profiles")

    args = parser.parse_args()

    with np.load(args.analysis[0]) as fh:
        ES_1T = fh['EIGENSOURCES'].T

    with np.load(args.analysis[1]) as fh:
        PROJECTION = np.matmul(ES_1T, fh['EIGENSOURCES'])
        del ES_1T

    BEST_MATCH = np.argmax(abs(PROJECTION), axis=1)
    SIGN = np.sign(PROJECTION[range(len(BEST_MATCH)),
                              BEST_MATCH]).reshape(1, 1, 1, -1)

    with np.load(args.eigensources[1]) as fh:
        CSD = 0.5 * SIGN * fh['CSD'].take(BEST_MATCH, axis=-1)

    with np.load(args.eigensources[0]) as fh:
        kwargs = {k: fh[k] for k in 'XYZ'}
        CSD += 0.5 * fh['CSD']

    np.savez_compressed(args.output,
                        CSD=CSD,
                        **kwargs)
