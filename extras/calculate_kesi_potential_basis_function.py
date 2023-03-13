#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy.integrate as si
# import pandas as pd

import _fast_reciprocal_reconstructor as frr
import _common_new as common


class Electrode(object):
    def __init__(self, filename):
        """
        Parameters
        ----------

        filename : str
            Path to the sampled correction potential.
        """
        self.filename = filename
        with np.load(filename) as fh:
            self.SAMPLING_GRID = [fh[c].flatten() for c in "XYZ"]
            self.x, self.y, self.z = fh["LOCATION"]
            self.base_conductivity = fh["BASE_CONDUCTIVITY"]

    def correction_leadfield(self, X, Y, Z):
        """
        Correction of the leadfield of the electrode
        for violation of kCSD assumptions

        Parameters
        ----------
        X, Y, Z : np.array
            Coordinate matrices of the same shape.
        """
        with np.load(self.filename) as fh:
            return self._correction_leadfield(fh["CORRECTION_POTENTIAL"],
                                              [X, Y, Z])

    def _correction_leadfield(self, SAMPLES, XYZ):
        # if XYZ points are in nodes of the sampling grid,
        # no time-consuming interpolation is necessary
        return SAMPLES[self._sampling_grid_indices(XYZ)]

    def _sampling_grid_indices(self, XYZ):
        return tuple(np.searchsorted(GRID, COORD)
                     for GRID, COORD in zip(self.SAMPLING_GRID, XYZ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate values of kCSD potential basis functions at electrode.")
    parser.add_argument("-o", "--output",
                        metavar="<output>",
                        dest="output",
                        help="output directory")
    parser.add_argument("-i", "--input",
                        metavar="<input>",
                        dest="input",
                        help="input directory")
    parser.add_argument("-c", "--centroids",
                        required=True,
                        metavar="<centroids.npz>",
                        help="centroids grid with mask")
    parser.add_argument("-s", "--source",
                        required=True,
                        metavar="<source.json>",
                        help="definition of shape of CSD basis function")
    parser.add_argument("names",
                        metavar="<electrode name>",
                        nargs="+",
                        help="names of electrodes")

    args = parser.parse_args()


    with np.load(args.centroids) as fh:
        CENTROID_XYZ = [fh[c] for c in ["X", "Y", "Z"]]
        CENTROID_MASK = fh["MASK"]

    for name in args.names:
        electrode = Electrode(os.path.join(args.input,
                              f'{name}.npz'))

        model_src = common.SphericalSplineSourceKCSD.fromJSON(
                                       open(args.source),
                                       conductivity=electrode.base_conductivity)

        d_xyz = np.array([(A[-1] - A[0]) / (len(A) - 1)
                          for A in electrode.SAMPLING_GRID])

        _ns = np.ceil(model_src.radius / d_xyz)
        romberg_ks = 1 + np.ceil(np.log2(_ns)).astype(int)

        romberg_weights = tuple(si.romb(np.identity(2 ** k + 1)) * 2.0 ** -k
                                for k in romberg_ks)

        LF_XYZ = [A[(A >= C.min() - model_src.radius)
                    & (A <= C.max() + model_src.radius)]
                  for A, C in zip(electrode.SAMPLING_GRID,
                                  CENTROID_XYZ)]

        convolver = frr.Convolver(LF_XYZ, LF_XYZ)

        CENTROIDS_IN_SRC = [np.isin(C.flatten(), S)
                            for S, C in zip(convolver.SRC_GRID, CENTROID_XYZ)]

        for c, IDX in zip("XYZ", CENTROIDS_IN_SRC):
            if not IDX.all():
                logger.warning(f"{(~IDX).sum()} centroid grid nodes missing along the {c} axis")

        SRC_IN_CENTROIDS = [np.isin(S.flatten(), C)
                            for S, C in zip(convolver.SRC_GRID, CENTROID_XYZ)]

        SRC_MASK = np.zeros(convolver.shape('SRC'),
                            dtype=bool)
        SRC_MASK[np.ix_(*SRC_IN_CENTROIDS)] = CENTROID_MASK[
                                                      np.ix_(*CENTROIDS_IN_SRC)]

        convolver_interface = frr.ConvolverInterfaceIndexed(convolver,
                                                            model_src.csd,
                                                            romberg_weights,
                                                            SRC_MASK)

        pae = frr.PAE_AnalyticalCorrectedNumerically(
                                                  convolver_interface,
                                                  potential=model_src.potential)

        with pae:
            np.savez_compressed(os.path.join(args.output,
                                             f"{name}.npz"),
                                POTENTIALS=(pae(electrode)),
                                X=electrode.x,
                                Y=electrode.y,
                                Z=electrode.z)
