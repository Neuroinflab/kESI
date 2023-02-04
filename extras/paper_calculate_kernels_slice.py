#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import scipy.integrate as si
import pandas as pd

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
            self.SAMPLING_GRID = [fh[c] for c in 'XYZ']
            self.x, self.y, self.z = fh['LOCATION']
            self.base_conductivity = fh['BASE_CONDUCTIVITY']

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
            return self._correction_leadfield(fh['CORRECTION_POTENTIAL'],
                                              [X, Y, Z])

    def _correction_leadfield(self, SAMPLES, XYZ):
        # if XYZ points are in nodes of the sampling grid,
        # no time-consuming interpolation is necessary
        return SAMPLES[self._sampling_grid_indices(XYZ)]

    def _sampling_grid_indices(self, XYZ):
        return tuple(np.searchsorted(GRID, COORD)
                     for GRID, COORD in zip(self.SAMPLING_GRID, XYZ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create 4SM kernels.')
    parser.add_argument('-o', '--output',
                        metavar='<output>',
                        dest='output',
                        help='output directory')
    parser.add_argument('-i', '--input',
                        metavar='<input>',
                        dest='input',
                        help='input directory')
    parser.add_argument('-k',
                        metavar='<k>',
                        type=int,
                        default=6,
                        dest='k',
                        help='K parameter of the Romberg method')
    parser.add_argument('--thickness',
                        type=float,
                        dest='h',
                        metavar='<thickness>',
                        help='slice thickness in meters (and length of an edge of the cubic volume of interest)',
                        default=3e-4)
    parser.add_argument('electrodes',
                        metavar='<electrode name>',
                        nargs='+',
                        help='ordered names of electrodes')

    args = parser.parse_args()

    electrodes = [Electrode(os.path.join(args.input,
                                         f'{name}.npz'))
                  for name in args.electrodes]

    ELECTRODES = []
    for name, electrode in zip(args.electrodes, electrodes):
        ELECTRODES.append({'NAME': name,
                           'X': electrode.x,
                           'Y': electrode.y,
                           'Z': electrode.z})
    ELECTRODES = pd.DataFrame(ELECTRODES)
    ELECTRODES.to_csv(os.path.join(args.output,
                                   'electrodes.csv'),
                      index=False)

    electrode = electrodes[0]

    XX, YY, ZZ = electrode.SAMPLING_GRID

    ROMBERG_K = args.k

    dx = (XX[-1] - XX[0]) / (len(XX) - 1)
    SRC_R_MAX = (2 ** (ROMBERG_K - 1)) * dx
    ROMBERG_N = 2 ** ROMBERG_K + 1

    H = args.h
    H_Y = H / 4
    X = XX
    Y = YY[abs(YY) <= H_Y + SRC_R_MAX + dx]
    Z = ZZ

    convolver = frr.Convolver([X, Y, Z],
                              [X, Y, Z])

    sd = SRC_R_MAX / 3

    def source(x, y, z):
        return common.SphericalSplineSourceKCSD(x, y, z,
                                                [sd, 3 * sd],
                                                [[1],
                                                 [0,
                                                  2.25 / sd,
                                                  -1.5 / sd ** 2,
                                                  0.25 / sd ** 3]],
                                                electrode.base_conductivity)

    model_src = source(0, 0, 0)

    SRC_MASK = ((abs(convolver.SRC_X) < abs(convolver.SRC_X.max()) - SRC_R_MAX)
                 & (abs(convolver.SRC_Y) <= H_Y)
                 & ((convolver.SRC_Z > SRC_R_MAX)
                    & (convolver.SRC_Z < H - SRC_R_MAX)))

    np.savez_compressed(os.path.join(args.output,
                                     f'src_mask.npz'),
                        MASK=SRC_MASK,
                        X=convolver.SRC_X,
                        Y=convolver.SRC_Y,
                        Z=convolver.SRC_Z)

    ROMBERG_WEIGHTS = si.romb(np.identity(ROMBERG_N)) * 2 ** -ROMBERG_K

    convolver_interface = frr.ConvolverInterfaceIndexed(convolver,
                                                        model_src.csd,
                                                        ROMBERG_WEIGHTS,
                                                        SRC_MASK)

    CSD_MASK = np.ones(convolver.shape('CSD'),
                       dtype=bool)

    kernel_constructor = frr.KernelConstructor()
    kernel_constructor.create_crosskernel = frr.CrossKernelConstructor(convolver_interface,
                                                                       CSD_MASK)

    del CSD_MASK

    paes = {'kCSD': frr.PAE_Analytical(
                            convolver_interface,
                            potential=model_src.potential),
            'kESI': frr.PAE_AnalyticalCorrectedNumerically(
                            convolver_interface,
                            potential=model_src.potential),
            }

    for method, pae in paes.items():
        PHI = kernel_constructor.create_base_images_at_electrodes(electrodes,
                                                                  pae)

        np.savez_compressed(os.path.join(args.output,
                                         f'{method}_phi.npz'),
                            PHI=PHI)

        KERNEL = kernel_constructor.create_kernel(PHI)

        np.savez_compressed(os.path.join(args.output,
                                         f'{method}_kernel.npz'),
                            KERNEL=KERNEL)

        CROSSKERNEL = kernel_constructor.create_crosskernel(PHI)
        crosskernel_shape = convolver.shape('CSD') + (-1,)
        np.savez_compressed(os.path.join(args.output,
                                         f'{method}_crosskernel.npz'),
                            CROSSKERNEL=CROSSKERNEL.reshape(crosskernel_shape),
                            X=convolver.CSD_X,
                            Y=convolver.CSD_Y,
                            Z=convolver.CSD_Z)
        del CROSSKERNEL

        _U, _S, _V = np.linalg.svd(PHI,
                                   full_matrices=False,
                                   compute_uv=True)
        del PHI

        np.savez_compressed(os.path.join(args.output,
                                         f'{method}_analysis.npz'),
                            EIGENVALUES=np.square(_S),
                            EIGENSOURCES=_U,
                            LAMBDAS=_S,
                            EIGENVECTORS=_V.T)
        del _S, _V

        EIGENSOURCES = np.full(convolver.shape('CSD') + _U.shape[1:],
                               np.nan)
        _SRC = np.zeros(convolver.shape('SRC'))
        for i, _SRC[SRC_MASK] in enumerate(_U.T):
            EIGENSOURCES[:, :, :, i] = convolver.base_weights_to_csd(
                                            _SRC,
                                            model_src.csd,
                                            (ROMBERG_N,) * 3)
        del _U, _SRC

        np.savez_compressed(os.path.join(args.output,
                                         f'{method}_eigensources.npz'),
                            CSD=EIGENSOURCES,
                            X=convolver.CSD_X,
                            Y=convolver.CSD_Y,
                            Z=convolver.CSD_Z)
        del EIGENSOURCES
