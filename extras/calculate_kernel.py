#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create kernel.")
    parser.add_argument("-o", "--output",
                        required=True,
                        metavar="<output>",
                        dest="output",
                        help="output directory")
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
                PHI = np.full((len(COL), len(ELECTRODES)), np.nan)

            PHI[:, i] = COL

    np.savez_compressed(os.path.join(args.output,
                                     "phi.npz"),
                        PHI=PHI)

    np.savez_compressed(os.path.join(args.output,
                                     "kernel.npz"),
                        KERNEL=np.matmul(PHI.T, PHI))

    _U, _S, _V = np.linalg.svd(PHI,
                               full_matrices=False,
                               compute_uv=True)
    del PHI

    np.savez_compressed(os.path.join(args.output,
                                     "analysis.npz"),
                        EIGENVALUES=np.square(_S),
                        EIGENSOURCES=_U,
                        LAMBDAS=_S,
                        EIGENVECTORS=_V.T)
