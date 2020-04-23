"""
@author: mbejtka
"""

import numpy as np
import pandas as pd
import itertools
import time

from kesi._verbose import (VerboseFFR,
                           LinearMixture,
                           LoadableVerboseFFR, _CrossKernelReconstructor)
from kesi._engine import _LinearKernelSolver

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
print('PARALLEL_AVAILABLE: ', PARALLEL_AVAILABLE)




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

def sigmoid_mean(error):
    '''
    Calculates sigmoidal mean across errors of reconstruction for many
    different sources - used for error maps.

    Parameters
    ----------
    error: numpy array
    Normalized point error of reconstruction.

    Returns
    -------
    error_mean: numpy array
    Sigmoidal mean error of reconstruction.
    error_mean -> 1    - very poor reconstruction
    error_mean -> 0    - perfect reconstruction
    '''
    sig_error = 2*(1./(1 + np.exp((-error))) - 1/2.)
    error_mean = np.mean(sig_error, axis=0)
    return error_mean


def source_scanning(sources, reconstructor, measurement_manager, measurement_manager_basis, EST_X, EST_Y, EST_Z, filename='reconstruction_error.npz'):
    point_error = []
    all_potentials = []
    all_true_csd = []
    all_est_csd = []
    for i in range(len(sources)):
        potential = measurement_manager.probe(sources[i])
        true_csd = measurement_manager_basis.probe(sources[i])
        approximator = reconstructor(potential)
        est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
        point_error.append(calculate_point_error(true_csd, est_csd))
        all_potentials.append(potential)
        all_true_csd.append(true_csd)
        all_est_csd.append(est_csd)
    point_error = np.array(point_error)
    error_mean = np.mean(point_error, axis=0) # sigmoid_mean(point_error)
    np.savez_compressed(filename,
    			POTS = potential,
    			TRUE_CSD = true_csd,
    			EST_CSD = est_csd,
    			ERROR = point_error,
    			ERROR_MEAN = error_mean)
    return error_mean


def make_reconstruction(source, reconstructor, measurement_manager, measurement_manager_basis, EST_X, EST_Y, EST_Z):
    pots = measurement_manager.probe(source)
    true_csd = measurement_manager_basis.probe(source)
    approximator = reconstructor(pots)
    est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
    error = calculate_point_error(true_csd, est_csd)
    return pots, true_csd, est_csd, error


def make_reconstruction_ck(pots, source, load_rec, cross_reconstructor, measurement_manager, measurement_manager_basis, regularization_parameter):
    true_csd = measurement_manager_basis.probe(source)
    rp = cross_validate(load_rec, pots, regularization_parameter)
    est_csd = cross_reconstructor(pots, rp)
    error = calculate_point_error(true_csd, est_csd)
    return true_csd, est_csd, error, rp


def source_scanning_parallel(potential, sources, load_rec, reconstructor, measurement_manager, measurement_manager_basis, regularization_parameter=0, filename='reconstruction_error.npz'):
    results = Parallel(n_jobs=NUM_CORES)(delayed(make_reconstruction_ck)
                                         (pots, source, load_rec, reconstructor,
                                          measurement_manager, measurement_manager_basis,
                                          regularization_parameter)
                                         for pots, source in zip(potential, sources))
    true_csd = np.array([item[0] for item in results])
    est_csd = np.array([item[1] for item in results])
    error = np.array([item[2] for item in results])
    rp = np.array([item[3] for item in results])
    error_mean = np.mean(error, axis=0) # sigmoid_mean(error)
    np.savez_compressed(filename,
    			TRUE_CSD = true_csd,
    			EST_CSD = est_csd,
    			ERROR = error,
    			ERROR_MEAN = error_mean,
                CV_RP = rp)
    return error_mean, true_csd, est_csd, error, rp


def cross_validate(reconstructor, measurements, regularization_parameters):
    errors = np.zeros(regularization_parameters.size)
    for rp_idx, rp in enumerate(regularization_parameters):
#        print('Cross validating regularization parameter :', rp)
        errors[rp_idx] = np.linalg.norm(reconstructor.leave_one_out_errors(measurements, rp))
    err_idx = np.where(errors == np.min(errors))
    cv_rp = regularization_parameters[err_idx][0]
#    print('CV_rp :', cv_rp)
    return cv_rp


start_time = time.time()    
MESHFILE = '/home/mbejtka/Data_Kuba/four_spheres_gaussian_1000_deg_1.npz'
factory = SomeSphereGaussianSourceFactory3D(MESHFILE)
print("Loading data --- %s seconds ---" % (time.time() - start_time))

dst = factory.R[1] - factory.R[0]
sources = [factory(r, altitude, azimuth)
           for altitude, azimuth in altitude_azimuth_mesh(-np.pi/2,
                                                          dst/factory.scalp_radius)
           for r in factory.R]
print('Number of sources: ', len(sources))
print("Sources --- %s seconds ---" % (time.time() - start_time))


# Determine positions of electrodes
theta, phi, r = np.meshgrid(np.linspace(-0.5*np.pi, 0.5*np.pi, 15),
                           np.linspace(0, 2*np.pi, 15),
                           [factory.R.max()])
ELE_X = r*np.cos(theta)*np.cos(phi)
ELE_Y = r*np.cos(theta)*np.sin(phi)
ELE_Z = r*np.sin(theta)
ELECTRODES = pd.DataFrame({'X': ELE_X.flatten(),
                           'Y': ELE_Y.flatten(),
                           'Z': ELE_Z.flatten()})

# Estimating points    
r = factory.scalp_radius
EST_X, EST_Y, EST_Z = np.meshgrid(np.linspace(-r, r, 30),
                                  np.linspace(-r, r, 30),
                                  np.linspace(-r, r, 30))
inside_sphere = np.array(np.where(EST_X.flatten()**2 + EST_Y.flatten()**2 + EST_Z.flatten()**2 <=r**2))
EST_X = EST_X.flatten()[inside_sphere[0]]
EST_Y = EST_Y.flatten()[inside_sphere[0]]
EST_Z = EST_Z.flatten()[inside_sphere[0]]
EST_POINTS =pd.DataFrame({'X': EST_X.flatten(),
                          'Y': EST_Y.flatten(),
                          'Z': EST_Z.flatten()})
    
measurement_manager = MeasurementManager(ELECTRODES, space='potential')
measurement_manager_basis = MeasurementManager(EST_POINTS, space='csd')

# Create reconstructor
reconstructor_filename = 'SavedReconstructor_four_spheres_1000_deg_1.npz'
#reconstructor = VerboseFFR(sources, measurement_manager)
#reconstructor.save(reconstructor_filename)
#print("Reconstructor --- %s seconds ---" % (time.time() - start_time))

factoryCSD = SomeSphereGaussianSourceFactoryOnlyCSD(MESHFILE)
dst = factoryCSD.R[1] - factoryCSD.R[0]
sourcesCSD = [factoryCSD(r, altitude, azimuth)
              for altitude, azimuth in altitude_azimuth_mesh(-np.pi/2,
                                                          dst/factory.scalp_radius)
              for r in factoryCSD.R]
# Load saved reconstructor
loadable_reconstructor = LoadableVerboseFFR(reconstructor_filename, sourcesCSD, measurement_manager)
kernel = loadable_reconstructor.kernel
cross_kernel = loadable_reconstructor.get_kernel_matrix(measurement_manager_basis)
cross_reconstructor = _CrossKernelReconstructor(_LinearKernelSolver(kernel), cross_kernel)

potential = [measurement_manager.probe(source) for source in sources]
filename = 'four_spheres_1000_deg_1_rp_cv.npz'



rps = np.logspace(-1, -15, 7, base=10.)
#est_csd = cross_reconstructor(potential[0], cross_validate(loadable_reconstructor, potential[0], rps))
#estimation_error = loadable_reconstructor.leave_one_out_errors(potential[0], 0.001)

error_time = time.time()
if PARALLEL_AVAILABLE:
    source_scanning_error, true_csd, est_csd, error, rp = source_scanning_parallel(potential, sourcesCSD, loadable_reconstructor, cross_reconstructor, measurement_manager, measurement_manager_basis, regularization_parameter=rps, filename=filename)
else:
    source_scanning_error = source_scanning(sources, reconstructor, measurement_manager,
                                            measurement_manager_basis, EST_X, EST_Y, EST_Z,
                                            filename=filename)
print("Error --- %s seconds ---" % (time.time() - error_time))
print("Total time --- %s seconds ---" % (time.time() - start_time))

