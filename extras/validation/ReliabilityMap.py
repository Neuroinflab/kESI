"""
@author: mbejtka
"""

import numpy as np
import pandas as pd
import itertools
import kesi
import scipy

from validate_properties import (ValidateKESI,
                                 MeasurementManager)

from FEM.fem_sphere_gaussian import SomeSphereGaussianSourceFactory3D


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
    epsilon = np.finfo(np.float64).eps
    point_error = np.linalg.norm(true_csd.reshape(true_csd.size, 1) -
                                 est_csd.reshape(est_csd.size, 1), axis=1)
    point_error /= np.linalg.norm(true_csd.reshape(true_csd.size, 1),
                                  axis=1) + \
                                  epsilon*np.max(np.linalg.norm(true_csd.reshape(true_csd.size, 1), axis=1))
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


def source_scanning(sources, reconstructor, measurement_manager, measurement_manager_basis, EST_X, EST_Y, EST_Z):
    point_error = []
    for i in range(len(sources)):
        potential = measurement_manager.probe(sources[i])
        true_csd = measurement_manager_basis.probe(sources[i])
        approximator = reconstructor(potential)
        est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
        point_error.append(calculate_point_error(true_csd, est_csd))
    point_error = np.array(point_error)
    error_mean = sigmoid_mean(point_error)
    return error_mean

    
factory = SomeSphereGaussianSourceFactory3D('/home/mbejtka/Data_Kuba/'
                                          'one_sphere_gaussian_0062_deg_1.npz')
Altitude = list(np.linspace(-0.5*np.pi, 0.5*np.pi, 40))
Azimuth = list(np.linspace(0, 2*np.pi, 40))
sources = all_sources(factory.R[::2], Altitude, Azimuth)

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

measurement_manager = MeasurementManager(ELECTRODES, space='potential')

reconstructor = ValidateKESI(sources, measurement_manager)

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

measurement_manager_basis = MeasurementManager(EST_POINTS, space='csd')


source_scanning_error = source_scanning(sources, reconstructor, measurement_manager, measurement_manager_basis, EST_X, EST_Y, EST_Z)