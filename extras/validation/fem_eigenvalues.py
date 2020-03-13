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

from FEM.fem_finite_slice_gaussian import FiniteSliceGaussianSourceFactory


def all_sources(xs, ys, zs):
    return [factory(x, y, z)
            for x, y, z in itertools.product(xs, ys, zs)]


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.dot(np.dot(csd, orthn), orthn.T) 


def calculate_diff(csd, projection):
    return np.abs(csd - projection)/np.max(csd)
    
    
factory = FiniteSliceGaussianSourceFactory('/home/mbejtka/FEM/kESI/extras/FEM/'
                                           'finite_slice_small_gaussian_0125_deg_2.npz')
# X = factory.X
# sources = all_sources(X + list(np.array(-1) * X), X, X + list(np.array(-1) * X))
sources = all_sources(factory.X_CENTROID, factory.Y_CENTROID, factory.Z_CENTROID)

ELE_X, ELE_Y, ELE_Z = np.meshgrid(np.linspace(-2.5e-04, 2.5e-04, 5),
                                  [3.0e-04],
                                  np.linspace(-2.5e-04, 2.5e-04, 5))
ELECTRODES = pd.DataFrame({'X': ELE_X.flatten(),
                           'Y': ELE_Y.flatten(),
                           'Z': ELE_Z.flatten()})
measurement_manager = MeasurementManager(ELECTRODES, space='potential')

reconstructor = ValidateKESI(sources, measurement_manager)

EST_X, EST_Y, EST_Z = np.meshgrid(np.linspace(-3.0e-04, 3.0e-04, 50),
                                  np.linspace(0, 3.0e-04, 10),
                                  np.linspace(-3.0e-04, 3.0e-04, 50))
EST_POINTS =pd.DataFrame({'X': EST_X.flatten(),
                          'Y': EST_Y.flatten(),
                          'Z': EST_Z.flatten()})
measurement_manager_basis = MeasurementManager(EST_POINTS, space='csd')
eigensources = reconstructor._eigensources(measurement_manager_basis)

potential_0_source = measurement_manager.probe(sources[100])
source0 = measurement_manager_basis.probe(sources[100])
vmax = np.max(abs(source0))
mlab.figure('Source100')
mlab.points3d(EST_X, EST_Y, EST_Z, source0.reshape(10, 50, 50), colormap='bwr',
              vmax=vmax, vmin=-vmax)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

approximator = reconstructor(potential_0_source)
# approximator = reconstructor(potential_0_source, regularization_parameter=10)
est_csd = approximator.csd(EST_X, EST_Y, EST_Z)
vmax = np.max(abs(est_csd))
mlab.figure('Reconstruction 100')
mlab.points3d(EST_X, EST_Y, EST_Z, est_csd, colormap='bwr',
              vmax=vmax, vmin=-vmax)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

a0 = eigensources[:, 0].reshape(10, 50, 50)
a1 = eigensources[:, 1].reshape(10, 50, 50)
a2 = eigensources[:, 2].reshape(10, 50, 50)
a3 = eigensources[:, 3].reshape(10, 50, 50)
# mlab.volume_slice(a1, colormap='RdBu')
# mlab.contour3d(a1, colormap='RdBu')

# mlab.pipeline.volume(mlab.pipeline.scalar_field(a1))
mlab.figure('1st eigensource')
mlab.points3d(EST_X, EST_Y, EST_Z, a0, colormap='bwr',
              vmax=np.max(abs(a0)), vmin=-np.max(abs(a0)))
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')
mlab.figure('2nd eigensource')
mlab.points3d(EST_X, EST_Y, EST_Z, a1, colormap='bwr',
              vmax=np.max(abs(a1)), vmin=-np.max(abs(a1)))
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')
mlab.figure('3rd eigensource')
mlab.points3d(EST_X, EST_Y, EST_Z, a2, colormap='bwr',
              vmax=np.max(abs(a2)), vmin=-np.max(abs(a2)))
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

projection = csd_into_eigensource_projection(source0, eigensources)
vmax = np.max(abs(projection))
mlab.figure('Projection 100')
mlab.points3d(EST_X, EST_Y, EST_Z, projection.reshape(10, 50, 50),
              colormap='bwr', vmax=vmax, vmin=-vmax)
mlab.colorbar(nb_labels=3)
mlab.xlabel('X')
mlab.ylabel('Y')

err = calculate_diff(source0, projection)
mlab.figure('Error 100')
mlab.points3d(EST_X, EST_Y, EST_Z, err.reshape(10, 50, 50), colormap='Greys')
mlab.colorbar()
mlab.xlabel('X')
mlab.ylabel('Y')

anihilation = calculate_diff(source0.reshape(10, 50, 50), est_csd)
mlab.figure('Error 100')
mlab.points3d(EST_X, EST_Y, EST_Z, anihilation, colormap='Greys')
mlab.colorbar()
mlab.xlabel('X')
mlab.ylabel('Y')

a = reconstructor.eigenvalues
plt.figure()
plt.title('Eigenvalues')
plt.plot(a, '.')
plt.yscale('log')
plt.show()
