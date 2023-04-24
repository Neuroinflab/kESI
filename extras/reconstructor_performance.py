#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from common import altitude_azimuth_mesh

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

def all_sources(factory, rs, altitudes, azimuths):
    return [factory(r, altitude, azimuth)
            for r, altitude, azimuth in itertools.product(rs, altitudes, azimuths)]

def cross_kernel_estimation(loadable_reconstructor, measurement_manager_basis,
                            potential, regularization_parameter=0):
    kernel = loadable_reconstructor.kernel
    cross_kernel = loadable_reconstructor.get_kernel_matrix(measurement_manager_basis)
    cross_reconstructor = _CrossKernelReconstructor(_LinearKernelSolver(kernel), cross_kernel)
    est_csd = cross_reconstructor(potential, regularization_parameter)
    return est_csd

def verbose_estimation(reconstructor, potential, EST_X, EST_Y, EST_Z,
                       regularization_parameter=0):
    approximator = reconstructor(potential, regularization_parameter)
    est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
    return est_csd

MESHFILE = '/home/mbejtka/Data_Kuba/one_sphere_gaussian_1000_deg_1.npz'
factory = SomeSphereGaussianSourceFactory3D(MESHFILE)

dst = factory.R[1] - factory.R[0]
sources = [factory(r, altitude, azimuth)
           for altitude, azimuth in altitude_azimuth_mesh(-np.pi/2,
                                                          dst/factory.scalp_radius)
           for r in factory.R]

# Electrodes
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
reconstructor_filename = 'SavedReconstructor_one_sphere_1000_deg_1.npz'
reconstructor = VerboseFFR(sources, measurement_manager)
reconstructor.save(reconstructor_filename)

# Generate ground truth (true_csd)
#factory2 = SomeSphereGaussianSourceFactory3D(MESHFILE)
true_csd = factory(factory.R[0], 0, 0)
potential = measurement_manager.probe(true_csd)

factory = SomeSphereGaussianSourceFactoryOnlyCSD(MESHFILE)
dst = factory.R[1] - factory.R[0]
sources = [factory(r, altitude, azimuth)
           for altitude, azimuth in altitude_azimuth_mesh(-np.pi/2,
                                                          dst/factory.scalp_radius)
           for r in factory.R]
# Load saved reconstructor
loadable_reconstructor = LoadableVerboseFFR(reconstructor_filename, sources, measurement_manager)
#
## Create cross kernel reconstructor
#kernel = loadable_reconstructor.kernel
#cross_kernel = loadable_reconstructor.get_kernel_matrix(measurement_manager_basis)
#cross_reconstructor = _CrossKernelReconstructor(_LinearKernelSolver(kernel), cross_kernel)
#
## Estimate solution
#est_csd = cross_reconstructor(potential, regularization_parameter=0)

# Save estimated data
#approximator_filename = 'Estimated_data_one_sphere_1000_deg_1.npz'
#np.savez_compressed(approximator_filename, CSD=est_csd,
#                    EST_X=EST_X,
#                    EST_Y=EST_Y,
#                    EST_Z=EST_Z, CROSS_KERNEL=cross_kernel) 
#utbs = abs(np.matmul(eigenvectors.T, potential))
t1 = time.time()
cross_csd = cross_kernel_estimation(loadable_reconstructor, measurement_manager_basis, potential, regularization_parameter=0)
print('cross kernel: ', time.time() - t1)
#%timeit cross_kernel_estimation(loadable_reconstructor, measurement_manager_basis, potential, regularization_parameter=0)                                      
##14min 43s ± 1.68 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

#%timeit cross_reconstructor(potential, regularization_parameter=0)      
##1.31 ms ± 35.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

t2 = time.time()
verbose_csd = verbose_estimation(reconstructor, potential, EST_X, EST_Y, EST_Z,
                                 regularization_parameter=0)
print('verbose: ', time.time() - t2)

#%timeit verbose_estimation(reconstructor, potential, EST_X, EST_Y, EST_Z)                                                                                      
##30.2 s ± 9.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)