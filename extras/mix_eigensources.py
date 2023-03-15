#!/usr/bin/env python
# coding: utf-8

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
