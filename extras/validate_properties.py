"""
@author: mbejtka
"""

import numpy as np
import pandas as pd

import kesi
from kesi._verbose import VerboseFFR
from _common_new import GaussianSourceKCSD3D


class ValidateKESI(VerboseFFR):
    def _eigensources(self, measurement_manager, regularization_parameter=0):
        kernel = self.kernel
        cross_kernel = self.cross_kernel(measurement_manager)
        self.eigenvalues, self.eigenvectors = self._evd(kernel,
                                                        regularization_parameter)
        return np.dot(cross_kernel, self.eigenvectors)

    def _evd(self, K, regularization_parameter=0):
        eigenvalues, eigenvectors = np.linalg.eigh(K + np.identity(K.shape[0])
                                                   * regularization_parameter)
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


H = 3e-4
standard_deviation = H / 16
conductivity = 0.3
X, Y = np.mgrid[0.05: 0.95: 10j,
                0.05: 0.95: 10j]
ELECTRODES = pd.DataFrame({'X': X.flatten(),
                           'Y': Y.flatten(),
                           })
forward = GaussianSourceKCSD3D(0, 0, 0, 3 * standard_deviation, conductivity)

src_X, src_Y = np.mgrid[0.:1.:95j,
                        0.:1.:95j]

sources = gaussian_source_factory_2d(src_X.flatten(),
                                     src_Y.flatten(),
                                     standard_deviation,
                                     conductivity)

measurement_manager = MeasurementManager(ELECTRODES)
reconstructor = ValidateKESI(sources, measurement_manager)
eigensources = reconstructor._eigensources(measurement_manager)
