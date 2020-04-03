"""
@author: mbejtka
"""

import numpy as np
import sys
import scipy
from numpy.linalg import LinAlgError
sys.path.append('../')

try:
    from api_stabilizer import VerboseFFR, MeasurementManagerBase, LinearMixture
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).
except (ImportError, SystemError, ValueError):
    from .api_stabilizer import VerboseFFR, MeasurementManagerBase, LinearMixture

from _common_new import GaussianSourceKCSD3D


class ValidateKESI(VerboseFFR):
    def _eigensources(self, measurement_manager_basis,
                      regularization_parameter=0):
        kernel = self.kernel
        cross_kernel = self.get_kernel_matrix(measurement_manager_basis)
        self.eigenvalues, self.eigenvectors = self._evd(kernel,
                                                        regularization_parameter)
        return np.dot(cross_kernel, self.eigenvectors)
    
    def _FieldEigenSource(self, EV):
        return LinearMixture(zip(self._field_components,
                                 np.dot(self._pre_kernel,
                                        EV)))
    
    def _field_eigensources(self, measurement_manager_basis,
                      regularization_parameter=0):
        kernel = self.kernel
        self.eigenvalues, self.eigenvectors = self._evd(kernel,
                                                        regularization_parameter)
        return [self._FieldEigenSource(EV)
                for EV in self.eigenvectors.T]

    def _evd(self, K, regularization_parameter=0):
        eigenvalues, eigenvectors = np.linalg.eigh(K + np.identity(K.shape[0])
                                                   * regularization_parameter)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def picard_plot(self, rhs, regularization_parameter):
        """
        Creates Picard plot according to Hansen's book.
        Parameters
        ----------
        rhs: numpy array
            Right-hand side of the linear equation.
        Raises
        ------
        LinAlgError
            If SVD computation does not converge.
        """
        try:
            u, s, v = np.linalg.svd(self.kernel + regularization_parameter *
                                    np.identity(self.kernel.shape[0]))
        except LinAlgError:
            raise LinAlgError('SVD is failing - try moving the electrodes'
                              'slightly')
        picard = np.zeros(len(s))
        picard_norm = np.zeros(len(s))
        for i, value in enumerate(s):
            picard[i] = abs(np.dot(u[:, i].T, rhs))
            picard_norm[i] = abs(np.dot(u[:, i].T, rhs))/value
        plt.figure(figsize=(10, 6))
        plt.plot(s, marker='.', label=r'$\sigma_{i}$')
        plt.plot(picard, marker='.', label='$|u(:, i)^{T}*rhs|$')
        plt.plot(picard_norm, marker='.',
                 label=r'$\frac{|u(:, i)^{T}*rhs|}{\sigma_{i}}$')
        plt.yscale('log')
        plt.legend()
        plt.title('Picard plot')
        plt.xlabel('i')
        plt.show()
        a = int(len(s) - int(np.sqrt(len(s)))**2)
        if a == 0:
            size = int(np.sqrt(len(s)))
        else:
            size = int(np.sqrt(len(s))) + 1
        fig, axs = plt.subplots(int(np.sqrt(len(s))),
                                size, figsize=(15, 13))
        axs = axs.ravel()
        beta = np.zeros(v.shape)
        fig.suptitle('vectors products of k_pot matrix')
        for i, value in enumerate(s):
            beta[i] = ((np.dot(u[:, i].T, rhs)/value) * v[i, :])
            axs[i].plot(beta[i, :], marker='.')
            axs[i].set_title(r'$vec_{'+str(i+1)+'}$')
        plt.show()

    # def _orthonormalize_matrix(self, matrix):
    #     orthn = scipy.linalg.orth(matrix)
    #     return orthn

    # def _csd_into_eigensource_projection(self, csd, eigensources):
    #     orthn = scipy.linalg.orth(eigensources)
    #     return np.dot(csd, orthn)
    
    # def _calculate_diff(self, a, b):
    #     return np.abs(a - b)


class MeasurementManager2d(MeasurementManagerBase):
    def __init__(self, ELECTRODES):
        self._ELECTRODES = ELECTRODES
        self.number_of_measurements = len(ELECTRODES)

    def probe(self, field):
        return field.potential(self._ELECTRODES.X,
                               self._ELECTRODES.Y,
                               0)


class MeasurementManager(MeasurementManagerBase):
    def __init__(self, ELECTRODES, space='potential'):
        self._space = space
        self._ELECTRODES = ELECTRODES
        self.number_of_measurements = len(ELECTRODES)

    def probe(self, field):
        return getattr(field, 
                       self._space)(self._ELECTRODES.X,
                                    self._ELECTRODES.Y,
                                    self._ELECTRODES.Z)

class MeasurementManagerFastP(MeasurementManagerBase):
    def __init__(self, ELECTRODES, space='potential'):
        self._space = space
        self._ELECTRODES = ELECTRODES
        self.number_of_measurements = len(ELECTRODES)

    def probe(self, field):
        return field.potential(self._ELECTRODES.X,
                               self._ELECTRODES.Y,
		               self._ELECTRODES.Z)


class MeasurementManagerFastC(MeasurementManagerBase):
    def __init__(self, ELECTRODES, space='potential'):
        self._space = space
        self._ELECTRODES = ELECTRODES
        self.number_of_measurements = len(ELECTRODES)

    def probe(self, field):
        return field.csd(self._ELECTRODES.X,
		         self._ELECTRODES.Y,
		         self._ELECTRODES.Z)


def gaussian_source_factory_2d(xs, ys, sd, conductivity):
    return [GaussianSourceKCSD3D(x, y, 0, sd, conductivity)
            for x, y in zip(xs, ys)]


def lanczos_source_factory_2d(xs, ys, sd, conductivity):
    return []


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

    measurement_manager = MeasurementManager2d(ELECTRODES)
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
    measurement_manager_basis = MeasurementManager2d(EST_POINTS)
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
