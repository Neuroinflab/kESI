#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np

import _fast_reciprocal_reconstructor as frr
import _common_new as common


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create volumetric crosskernel.")
    parser.add_argument("-o", "--output",
                        required=True,
                        metavar="<output.npz>",
                        dest="output",
                        help="volumetric crosskernel")
    parser.add_argument("-i", "--input",
                        metavar="<phi.npz>",
                        dest="input",
                        help="source transfer matrix")
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
                     for _d in convolver.steps()]
    fake_weights = [[None] * n for n in src_diameters]

    convolver_interface = frr.ConvolverInterfaceIndexed(convolver,
                                                        model_src.csd,
                                                        fake_weights,
                                                        SRC_MASK)

    crosskernel_constructor = frr.CrossKernelConstructor(
                                                 convolver_interface,
                                                 np.ones(convolver.shape('CSD'),
                                                         dtype=bool))

    with np.load(args.input) as fh:
        PHI = fh["PHI"]

    CROSSKERNEL = crosskernel_constructor(PHI)
    crosskernel_shape = convolver.shape('CSD') + (-1,)
    np.savez_compressed(os.path.join(args.output),
                        CROSSKERNEL=CROSSKERNEL.reshape(crosskernel_shape),
                        X=convolver.CSD_X,
                        Y=convolver.CSD_Y,
                        Z=convolver.CSD_Z)
