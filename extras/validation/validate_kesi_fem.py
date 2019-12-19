"""
@author: mbejtka
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import FEM.fem_gaussian as fem_gaussian
from validation.validate_properties import MeasurementManager, ValidateKESI


def all_gaussians(xs, ys, zs, sd, conductivity):
    return [gaussian_source_factory(x, y, z, sd, conductivity)
            for x, y, z in zip(xs, ys, zs)]


gaussian_source_factory = fem_gaussian.GaussianSourceFactory('eighth_of_sphere_gaussian_1000.npz')

std_dev = 0.08
conductivity = 0.3
N_src_X = 100
N_src_Y = 100

X, Y = np.meshgrid(np.linspace(0.025, 4.975, N_src_X),
                   np.linspace(0.025, 4.975, N_src_Y))
all_sources = all_gaussians(X.flatten(), Y.flatten(), np.zeros(N_src_X*N_src_Y), std_dev, conductivity)

ELE_X, ELE_Y, ELE_Z = np.meshgrid(np.linspace(0.25, 4.75, 10),
                                  np.linspace(0.25, 4.75, 10),
                                  [0])
ELECTRODES = pd.DataFrame({'X': ELE_X.flatten(),
                           'Y': ELE_Y.flatten(),
                           'Z': ELE_Z.flatten()})

measurement_manager = MeasurementManager(ELECTRODES, space='potential')

reconstructor = ValidateKESI(all_sources, measurement_manager)

EST_X, EST_Y, EST_Z = np.meshgrid(np.linspace(0.025, 4.975, N_src_X),
                                  np.linspace(0.025, 4.975, N_src_Y),
                                  [0])
EST_POINTS =pd.DataFrame({'X': EST_X.flatten(),
                          'Y': EST_Y.flatten(),
                          'Z': EST_Z.flatten()})
measurement_manager_basis = MeasurementManager(EST_POINTS, space='csd')
eigensources = reconstructor._eigensources(measurement_manager_basis)

fig = plt.figure(figsize=(18, 16))
heights = [1, 1, 1, 1]

gs = gridspec.GridSpec(4, 4, height_ratios=heights, hspace=0.6, wspace=0.5)
nr_plts = 16

for i in range(nr_plts):
    ax = fig.add_subplot(gs[i], aspect='equal')

    a = eigensources[:, i].reshape(N_src_X, N_src_Y, -1)
    cset = ax.contourf(EST_X[:, :, 0], EST_Y[:, :, 0], a[:, :, 0], cmap=cm.bwr)

    ax.text(0.5, 1.05, r"$\tilde{K} \cdot{v_{{%(i)d}}}$" % {'i': i+1},
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=15)

a = reconstructor.eigenvalues
plt.figure()
plt.plot(a, '.')
plt.yscale('log')
plt.show()