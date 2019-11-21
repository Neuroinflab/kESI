"""
@author: mbejtka
"""

import numpy as np
import sys
sys.path.append('../')

import kesi
from kesi._verbose import VerboseFFR
from _common_new import GaussianSourceKCSD3D


class ValidateKESI(VerboseFFR):
    def _eigensources(self, measurement_manager_basis,
                      regularization_parameter=0):
        kernel = self.kernel
        cross_kernel = self.cross_kernel(measurement_manager_basis)
        self.eigenvalues, self.eigenvectors = self._evd(kernel,
                                                        regularization_parameter)
        return np.dot(cross_kernel, self.eigenvectors)

    def _evd(self, K, regularization_parameter=0):
        eigenvalues, eigenvectors = np.linalg.eigh(K + np.identity(K.shape[0])
                                                   * regularization_parameter)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors


class MeasurementManager(kesi.MeasurementManagerBase):
    def __init__(self, ELECTRODES):
        self._ELECTRODES = ELECTRODES
        self.number_of_measurements = len(ELECTRODES)

    def probe(self, field):
        return field.potential(self._ELECTRODES.X,
                               self._ELECTRODES.Y,
                               0)


def gaussian_source_factory_2d(xs, ys, sd, conductivity):
    return [GaussianSourceKCSD3D(x, y, 0, sd, conductivity)
            for x, y in zip(xs, ys)]


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    H = 3e-4
    standard_deviation = H*100  # H / 16
    conductivity = 0.3
    X, Y = np.mgrid[0.05: 0.95: 10j,
                    0.05: 0.95: 10j]
    ELECTRODES = pd.DataFrame({'X': X.flatten(),
                               'Y': Y.flatten(),
                               })

    measurement_manager = MeasurementManager(ELECTRODES)
    src_X, src_Y = np.mgrid[0.:1.:100j,
                            0.:1.:100j]

    sources = gaussian_source_factory_2d(src_X.flatten(),
                                         src_Y.flatten(),
                                         standard_deviation,
                                         conductivity)

    reconstructor = ValidateKESI(sources, measurement_manager)

    est_X, est_Y = np.mgrid[0.:1.:100j,
                            0.:1.:100j]
    EST_POINTS = pd.DataFrame({'X': est_X.flatten(),
                               'Y': est_Y.flatten(),
                               })
    measurement_manager_basis = MeasurementManager(EST_POINTS)
    eigensources = reconstructor._eigensources(measurement_manager_basis)

    fig = plt.figure(figsize=(18, 16))
    heights = [1, 1, 1, 1]

    gs = gridspec.GridSpec(4, 4, height_ratios=heights, hspace=0.6, wspace=0.5)
    nr_plts = 16

    for i in range(nr_plts):
        ax = fig.add_subplot(gs[i], aspect='equal')

        a = eigensources[:, i].reshape(len(est_X), len(est_X), 1)
        cset = ax.contourf(est_X, est_Y, a[:, :, 0], cmap=cm.bwr)

        ax.text(0.5, 1.05, r"$\tilde{K} \cdot{v_{{%(i)d}}}$" % {'i': i+1},
                horizontalalignment='center', transform=ax.transAxes,
                fontsize=15)

    a = reconstructor.eigenvalues
    plt.figure()
    plt.plot(a)
    plt.yscale('log')
    plt.show()
