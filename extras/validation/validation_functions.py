import numpy as np
import pandas as pd
import scipy
import itertools
import time
import sys

from kesi._verbose import (VerboseFFR,
                           LinearMixture,
                           LoadableVerboseFFR)
from kesi._engine import _LinearKernelSolver
sys.path.append('..')
from FEM.fem_sphere_gaussian import (SomeSphereGaussianSourceFactory3D,
                                     SomeSphereGaussianSourceFactoryOnlyCSD)
from _common_new import altitude_azimuth_mesh

try:
    from joblib import Parallel, delayed
    import multiprocessing
    NUM_CORES = multiprocessing.cpu_count() - 1
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

MeasurementManagerBase = VerboseFFR.MeasurementManagerBase


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


def all_sources(r, altitude, azimuth):
    return [factory(x, y, z)
            for x, y, z in itertools.product(r, altitude, azimuth)]


def calculate_point_error(true_csd, est_csd):
    """
    Calculates normalized error of reconstruction at every point of
    estimation space separetly.

    Parameters
    ----------
    true_csd: numpy array
        Values of true csd at points of kCSD estimation.
    est_csd: numpy array
        CSD estimated with kCSD method.

    Returns
    -------
    point_error: numpy array
        Normalized error of reconstruction calculated separetly at every
        point of estimation space.
    """
    true_csd_r = true_csd.reshape(true_csd.size, 1)
    est_csd_r = est_csd.reshape(est_csd.size, 1)
    epsilon = np.linalg.norm(true_csd_r)/np.max(abs(true_csd_r))
    err_r = abs(est_csd_r/(np.linalg.norm(est_csd_r)) -
                true_csd_r/(np.linalg.norm(true_csd_r)))
    err_r *= epsilon
    point_error = err_r.reshape(true_csd.shape)
    return point_error


def calculate_rms(true_csd, est_csd):
    """
    Calculates normalized error of reconstruction.
    Parameters
    ----------
    true_csd: numpy array
        Values of true CSD at points of kCSD estimation.
    est_csd: numpy array
        CSD estimated with kCSD method.
    Returns
    -------
    rms: float
        Normalized error of reconstruction.
    """
    rms = np.linalg.norm((true_csd - est_csd))/(np.linalg.norm(true_csd))
    return rms


def calculate_rdm(true_csd, est_csd):
    """
    Calculates relative difference measure between reconstructed source and
    ground truth.
    Parameters
    ----------
    true_csd: numpy array
        Values of true CSD at points of kCSD estimation.
    est_csd: numpy array
        CSD estimated with kCSD method.
    Returns
    -------
    rdm: float
        Relative difference measure.
    """
    epsilon = np.finfo(np.float64).eps
    rdm = np.linalg.norm(est_csd/(np.linalg.norm(est_csd) + epsilon) -
                         true_csd/(np.linalg.norm(true_csd) + epsilon))
    return rdm


def calculate_rdm_point(true_csd, est_csd):
    rdm = abs(est_csd.reshape(est_csd.size, 1)/(np.linalg.norm(est_csd.reshape(est_csd.size, 1))) -
              true_csd.reshape(true_csd.size, 1)/(np.linalg.norm(true_csd.reshape(true_csd.size, 1))))
    rdm *= np.linalg.norm(true_csd.reshape(true_csd.size, 1))/np.max(abs(true_csd.reshape(true_csd.size, 1)))
    return rdm.reshape(true_csd.shape)


def calculate_mag(true_csd, est_csd):
    """
    Calculates magnitude ratio between reconstructed source and ground
    truth.
    Parameters
    ----------
    test_csd: numpy array
        Values of true CSD at points of kCSD estimation.
    est_csd: numpy array
        CSD estimated with kCSD method.
    Returns
    -------
    mag: float
        Magnitude ratio.
    """
    epsilon = np.finfo(np.float64).eps
    mag = np.linalg.norm(est_csd/(true_csd + epsilon))
    return mag


def calculate_mag_point(true_csd, est_csd):
    epsilon = np.max(abs(true_csd.reshape(true_csd.size, 1)))
    mag = abs(est_csd.reshape(est_csd.size, 1))/(abs(true_csd.reshape(true_csd.size, 1)) + epsilon)
    return mag.reshape(true_csd.shape)


def cross_validation(reconstructor, measurements, regularization_parameters):
    EE = np.zeros((regularization_parameters.size, np.array(measurements).shape[0]))
    for rp_idx, rp in enumerate(regularization_parameters):
        print('Cross validating regularization parameter :', rp)
        EE[rp_idx] = np.linalg.norm(reconstructor.leave_one_out_errors(np.array(measurements).T, rp), axis=0)
    indx_rp = np.argmin(EE, axis=0)
    return indx_rp, EE


def estimate_csd(reconstructor, measurements, regularization_parameters):
    indx_rp, EE = cross_validation(reconstructor, measurements, regularization_parameters)
    if np.array(measurements).shape[0] == np.array(measurements).size:
        indx_rp = indx_rp[0]
        EST_CSD = reconstructor(np.array(measurements).T, regularization_parameters[indx_rp]).T
        EE = EE[:, 0]
        lambd = regularization_parameters[indx_rp]
        print('CV_rp :', regularization_parameters[indx_rp])
    else:
        EST_CSD = np.zeros(((np.array(measurements).shape[0]), reconstructor._cross_kernel.shape[0]))
        lambd = np.zeros([(np.array(measurements).shape[0])])
        for i, rp in enumerate(regularization_parameters):
            lambd[indx_rp == i] = rp
            EST_CSD[indx_rp == i, :] = reconstructor(np.array(measurements)[indx_rp == i, :].T, rp).T
    return EST_CSD, indx_rp, EE, rp


def calculate_eigensources(kernel, cross_kernel, regularization_parameter=0):
    eigenvalues, eigenvectors = np.linalg.eigh(kernel + np.identity(kernel.shape[0])
                                               * regularization_parameter)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return np.matmul(cross_kernel, eigenvectors), eigenvalues, eigenvectors


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.matmul(np.matmul(csd, orthn), orthn.T)


def parallel_search(kernel, potential, lambdas, n_jobs=4):
    """Method for Parallel L-curve computation
    Parameters
    ----------
    kernel : np.array
    potential : list
    lambdas : list
    Returns
    -------
    modelnormseq : list
    residualseq : list
    """
    if PARALLEL_AVAILABLE:
        jobs = (delayed(L_model_fast)(kernel, potential, lamb, i)
                for i, lamb in enumerate(lambdas))
        modelvsres = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    else:
        # Please verify this!
        modelvsres = []
        for i, lamb in enumerate(lambdas):
            modelvsres.append(L_model_fast(kernel, potential, lamb, i))
    modelnormseq, residualseq = zip(*modelvsres)
    return modelnormseq, residualseq


def L_model_fast(kernel, potential, lamb, i):
    """Method for Fast L-curve computation
    Parameters
    ----------
    kernel : np.array
    potential : list
    lambd : list
    i : int
    Returns
    -------
    modelnorm : float
    residual : float
    """
    k_inv = np.linalg.inv(kernel + lamb*np.identity(kernel.shape[0]))
    beta_new = np.dot(k_inv, potential)
    V_est = np.dot(kernel, beta_new)
    modelnorm = np.einsum('i,j->i', beta_new.T, V_est)
    residual = np.linalg.norm(V_est - potential)
    modelnorm = np.max(modelnorm)
    return modelnorm, residual


def suggest_lambda(kernel):
    """Computes the lambda parameter range for regularization, 
    Used in Cross validation and L-curve
    Returns
    -------
    Lambdas : list
    """
    s, v = np.linalg.eigh(kernel)
    print(s)
    print('min lambda', 10**np.round(np.log10(s[0]), decimals=0))
    print('max lambda', str.format('{0:.4f}', np.std(np.diag(kernel))))
    return np.logspace(np.log10(s[0]), np.log10(np.std(np.diag(kernel))), 20)


def L_curve(kernel, potential, lambdas=None, n_jobs=1):
    """Method defines the L-curve.
    By default calculates L-curve over lambda,
    When no argument is passed, it takes
    lambdas = np.logspace(-10,-1,100,base=10)
    and Rs = np.array(self.R).flatten()
    otherwise pass necessary numpy arrays
    Parameters
    ----------
    L-curve plotting: default True
    lambdas : numpy array
    Rs : numpy array
    Returns
    -------
    curve_surf : post cross validation
    """
    if lambdas is None:
        print('No lambda given, using defaults')
        lambdas = suggest_lambda(kernel)
    else:
        lambdas = lambdas.flatten()
    lcurve_axis = np.zeros((2, len(lambdas)))
    curve_surf = np.zeros((len(lambdas)))
    suggest_lambda(kernel)
    #print('l-curve (all lambda): ', np.round(R, decimals=3))
    modelnormseq, residualseq = parallel_search(kernel, potential, lambdas,
                                                      n_jobs=n_jobs)
    norm_log = np.log(modelnormseq + np.finfo(np.float64).eps)
    res_log = np.log(residualseq + np.finfo(np.float64).eps)
    curveseq = res_log[0] * (norm_log - norm_log[-1]) + res_log * (norm_log[-1] - norm_log[0]) \
        + res_log[-1] * (norm_log[0] - norm_log)
    curve_surf = curveseq
    lcurve_axis[0, :] = norm_log
    lcurve_axis[1, :] = res_log
    lambd = lambdas[np.argmax(curve_surf)]
    #self.update_lambda(lambdas[np.argmax(self.curve_surf, axis=1)[best_R_ind]])
    print("Best lambda = ", lambd)
    return lambd, lcurve_axis, curve_surf, lambdas


def add_noise(potential, seed=0, level=10):
    """
    Adds Gaussian noise to potentials.
    Parameters
    ----------
    potential: numpy array
        Potentials at measurement points.
    seed: int
        Random seed generator.
        Default: 0.
    level: float, optional
        Noise level in percentage.
        Default: 10.
    Returns
    -------
    potential_noise: numpy array
        Potentials with added random Gaussian noise.
    """
    rstate = np.random.RandomState(seed)
    noise = 0.01*level*rstate.normal(0, np.std(potential),
                                     len(potential))
    potential_noise = potential + noise
    return potential_noise, noise


