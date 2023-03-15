#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np

import _fast_reciprocal_reconstructor as frr
import _common_new as common


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create volumetric eigensources.")
    parser.add_argument("-o", "--output",
                        required=True,
                        metavar="<output.npz>",
                        dest="output",
                        help="volumetric eigensources")
    parser.add_argument("-i", "--input",
                        required=True,
                        metavar="<analysis.npz>",
                        dest="input",
                        help="auxiliary kernel analysis matrices")
    parser.add_argument("-c", "--centroids",
                        required=True,
                        metavar="<centroids.npz>",
                        help="centroids grid with mask")
    parser.add_argument("-g", "--grid",
                        required=True,
                        metavar="<grid.npz>",
                        help="CSD grid")
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
        SRC_MASK = fh["MASK"]

    with np.load(args.grid) as fh:
        csd_grid = [fh[c] for c in args.coords]

    model_src = common.SphericalSplineSourceBase.fromJSON(open(args.source))

    convolver = frr.Convolver(centroids, csd_grid)

    src_diameters = [int(2 * np.floor(model_src.radius / _d)) + 1
                     for _d in convolver.steps("CSD")]

    with np.load(args.input) as fh:
        CANONICAL_ES = fh["EIGENSOURCES"]

    EIGENSOURCES = np.full(convolver.shape('CSD') + CANONICAL_ES.shape[1:],
                           np.nan)
    _SRC = np.zeros(convolver.shape('SRC'))
    for i, _SRC[SRC_MASK] in enumerate(CANONICAL_ES.T):
        EIGENSOURCES[:, :, :, i] = convolver.base_weights_to_csd(_SRC,
                                                                 model_src.csd,
                                                                 src_diameters)
    del CANONICAL_ES, _SRC

    np.savez_compressed(os.path.join(args.output),
                        CSD=EIGENSOURCES,
                        X=convolver.CSD_X,
                        Y=convolver.CSD_Y,
                        Z=convolver.CSD_Z)
