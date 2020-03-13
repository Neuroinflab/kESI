"""
@author: mbejtka
"""

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from mayavi import mlab
import kesi
import scipy

from validate_properties import (ValidateKESI,
                                 MeasurementManager)

from FEM.fem_sphere_gaussian import SomeSphereGaussianSourceFactory


def all_sources(r, altitude, azimuth):
    return [factory(x, y, z)
            for x, y, z in itertools.product(r, altitude, azimuth)]


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.dot(np.dot(csd, orthn), orthn.T) 

def calculate_diff(csd, projection):
    return np.abs(csd - projection)/np.max(csd)

    
factory = SomeSphereGaussianSourceFactory('/home/jdzik/FEM_soultions/'
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
mlab.figure('sources')
mlab.points3d(X, Y, Z, opacity=0.2, scale_mode='none')

theta, phi, r = np.meshgrid(np.linspace(-0.5*np.pi, 0.5*np.pi, 15),
                           np.linspace(0, 2*np.pi, 15),
                           [factory.R.max()])
ELE_X = r*np.cos(theta)*np.cos(phi)
ELE_Y = r*np.cos(theta)*np.sin(phi)
ELE_Z = r*np.sin(theta)
ELECTRODES = pd.DataFrame({'X': ELE_X.flatten(),
                           'Y': ELE_Y.flatten(),
                           'Z': ELE_Z.flatten()})
mlab.figure('Electrodes')
mlab.points3d(ELE_X, ELE_Y, ELE_Z)
measurement_manager = MeasurementManager(ELECTRODES, space='potential')

reconstructor = ValidateKESI(sources, measurement_manager)

EST_R, EST_ALTITUDE, EST_AZIMUTH = np.meshgrid(np.linspace(0, factory.R.max(), 64),
                                  list(np.linspace(-0.5*np.pi, 0.5*np.pi, 10)),
                                  list(np.linspace(0, 2*np.pi, 10)))
# EST_X = EST_R*np.cos(EST_ALTITUDE)*np.cos(EST_AZIMUTH)
# EST_Y = EST_R*np.cos(EST_ALTITUDE)*np.sin(EST_AZIMUTH)
# EST_Z = EST_R*np.sin(EST_ALTITUDE)
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
mlab.figure('regular grid')
mlab.points3d(EST_X, EST_Y, EST_Z)
# mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

measurement_manager_basis = MeasurementManager(EST_POINTS, space='csd')
eigensources = reconstructor._eigensources(measurement_manager_basis)

potential_0_source = measurement_manager.probe(sources[10])
source10 = measurement_manager_basis.probe(sources[10])
vmax = np.max(abs(source10))
mlab.figure('Source regular grid')
mlab.points3d(EST_X, EST_Y, EST_Z, source10.reshape(EST_X.shape), colormap='bwr',
              vmax=vmax, vmin=-vmax)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

a0 = eigensources[:, 0].reshape(EST_X.shape)
a1 = eigensources[:, 1].reshape(EST_X.shape)
a2 = eigensources[:, 2].reshape(EST_X.shape)
a3 = eigensources[:, 3].reshape(EST_X.shape)
visible_eigensources = eigensources[:, :8]

mlab.figure('1st eigensource regular grid')
mlab.points3d(EST_X, EST_Y, EST_Z, a0, colormap='bwr',
              vmax=np.max(abs(a0)), vmin=-np.max(abs(a0)), opacity=0.3, scale_mode='none', resolution=3)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')
mlab.figure('2nd eigensource')
mlab.points3d(EST_X, EST_Y, EST_Z, a1, colormap='bwr',
              vmax=np.max(abs(a1)), vmin=-np.max(abs(a1)), opacity=0.1, scale_mode='none')
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')
mlab.figure('3rd eigensource')
mlab.points3d(EST_X, EST_Y, EST_Z, a2, colormap='bwr',
              vmax=np.max(abs(a2)), vmin=-np.max(abs(a2)))
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

charge0 = np.sum(a0)
charge1 = np.sum(a1)
charge2 = np.sum(a2)
charge3 = np.sum(a3)

approximator = reconstructor(potential_0_source, regularization_parameter=100)
est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
vmax = np.max(abs(est_csd))
mlab.figure('Reconstruction 10 lambda 100 scale none')
mlab.points3d(EST_X, EST_Y, EST_Z, est_csd, colormap='bwr',
              vmax=vmax, vmin=-vmax, opacity=0.3, scale_mode='none')
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

projection = csd_into_eigensource_projection(source10, eigensources)
vmax = np.max(abs(projection))
mlab.figure('Projection 10')
mlab.points3d(EST_X, EST_Y, EST_Z, projection.reshape(EST_X.shape),
              colormap='bwr', vmax=vmax, vmin=-vmax)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

new_projection = csd_into_eigensource_projection(source10, visible_eigensources)
vmax = np.max(abs(new_projection))
mlab.figure('New Projection 10')
mlab.points3d(EST_X, EST_Y, EST_Z, new_projection.reshape(EST_X.shape),
              colormap='bwr', vmax=vmax, vmin=-vmax)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

err = calculate_diff(source10, projection)
mlab.figure('Error 10')
mlab.points3d(EST_X, EST_Y, EST_Z, err.reshape(EST_X.shape), colormap='Greys')
mlab.colorbar()
mlab.xlabel('X')
mlab.ylabel('Y')