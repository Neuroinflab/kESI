#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np

import _common_new as common


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
