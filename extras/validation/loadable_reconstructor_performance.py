#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mbejtka
"""
import numpy as np
import pandas as pd
import itertools
import kesi
import scipy
import time

from kesi._verbose import (VerboseFFR,
                           LinearMixture,
                           LoadableVerboseFFR)
from FEM.fem_sphere_gaussian import SomeSphereGaussianSourceFactory3D

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


start_time = time.time()    
factory = SomeSphereGaussianSourceFactory3D('/home/mbejtka/Data_Kuba/'
                                          'one_sphere_gaussian_0062_deg_1.npz')
print("Loading data --- %s seconds ---" % (time.time() - start_time))
Altitude = list(np.linspace(-0.5*np.pi, 0.5*np.pi, 40))
Azimuth = list(np.linspace(0, 2*np.pi, 40))
sources = all_sources(factory.R[::2], Altitude, Azimuth)
print("Sources --- %s seconds ---" % (time.time() - start_time))

R, altitude, azimuth = np.meshgrid(factory.R[::2],
                                   Altitude,
                                   Azimuth)
X = R*np.cos(altitude)*np.cos(azimuth)
Y = R*np.cos(altitude)*np.sin(azimuth)
Z = R*np.sin(altitude)

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

r = factory.scalp_radius
EST_X, EST_Y, EST_Z = np.meshgrid(np.linspace(-r, r, 20),
                                  np.linspace(-r, r, 20),
                                  np.linspace(-r, r, 20))
inside_sphere = np.array(np.where(EST_X.flatten()**2 + EST_Y.flatten()**2 + EST_Z.flatten()**2 <=r**2))
EST_X = EST_X.flatten()[inside_sphere[0]]
EST_Y = EST_Y.flatten()[inside_sphere[0]]
EST_Z = EST_Z.flatten()[inside_sphere[0]]
EST_POINTS =pd.DataFrame({'X': EST_X.flatten(),
                          'Y': EST_Y.flatten(),
                          'Z': EST_Z.flatten()})

t3 = time.time()
measurement_manager = MeasurementManager(ELECTRODES, space='potential')
print('MMF: ', time.time() - t3)
t4 = time.time()
measurement_manager_basis = MeasurementManager(EST_POINTS, space='csd')
print('MMF_basis: ', time.time() - t4)


rec_time = time.time()
reconstructor = VerboseFFR(sources, measurement_manager)
reconstructor_file = 'SavedReconstructor.npz'
reconstructor.save(reconstructor_file)
print("Reconstructor MM --- %s seconds ---" % (time.time() - rec_time))

loaded_reconstructor = LoadableVerboseFFR(reconstructor_file, sources, measurement_manager)
kernel = loaded_reconstructor.kernel

def estimation(source, reconstructor, measurement_manager,
               measurement_manager_basis, EST_X, EST_Y, EST_Z):
    potential = measurement_manager.probe(source)
    true_csd = measurement_manager_basis.probe(source)
    approximator = reconstructor(potential)
    est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
    return est_csd

st = time.time()
estimation(sources[0], reconstructor, measurement_manager,
           measurement_manager_basis, EST_X, EST_Y, EST_Z)
print("Approximator MM --- %s seconds ---" % (time.time() - st))

#from kesi._engine import _LinearKernelSolver
#potential = measurement_manager.probe(sources[0])
#est_csd = _LinearKernelSolver(kernel)
