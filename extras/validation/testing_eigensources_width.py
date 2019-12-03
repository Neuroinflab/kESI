"""
@author: mbejtka
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from validate_properties import (ValidateKESI,
                                 MeasurementManager,
                                 gaussian_source_factory_2d)


H = 3e-4
conductivity = 0.3
X, Y = np.mgrid[0.05: 0.95: 10j,
                0.05: 0.95: 10j]
ELECTRODES = pd.DataFrame({'X': X.flatten(),
                           'Y': Y.flatten(),
                           })

measurement_manager = MeasurementManager(ELECTRODES)
src_X, src_Y = np.mgrid[0.:1.:95j,
                        0.:1.:95j]
est_X, est_Y = np.mgrid[0.:1.:100j,
                        0.:1.:100j]
EST_POINTS = pd.DataFrame({'X': est_X.flatten(),
                           'Y': est_Y.flatten(),
                           })
measurement_manager_basis = MeasurementManager(EST_POINTS)
standard_deviation = [H/10, H/5, H, H*5, H*10, H*50, H*100, H*250]
reconstructor_list = []
eigensources_list = []
for i in range(len(standard_deviation)):
    sources = gaussian_source_factory_2d(src_X.flatten(),
                                         src_Y.flatten(),
                                         standard_deviation[i],
                                         conductivity)
    reconstructor = ValidateKESI(sources, measurement_manager)
    reconstructor_list.append(reconstructor)
    eigensources = reconstructor._eigensources(measurement_manager_basis)
    eigensources_list.append(eigensources)


plt.figure()
for i in range(len(standard_deviation)):
    plt.plot(reconstructor_list[i].eigenvalues, label=i)
plt.yscale('log')
plt.xlabel('Number of components')
plt.ylabel('Eigenvalues')
plt.legend()
plt.show()

###
fig = plt.figure(figsize=(18, 16))
heights = [1, 1, 1, 1]

gs = gridspec.GridSpec(4, 4, height_ratios=heights, hspace=0.6, wspace=0.5)
nr_plts = 16

for i in range(nr_plts):
    ax = fig.add_subplot(gs[i], aspect='equal')

    a = eigensources_list[7][:, i].reshape(len(est_X), len(est_X), 1)
    cset = ax.contourf(est_X, est_Y, a[:, :, 0], cmap=cm.bwr)

    ax.text(0.5, 1.05, r"$\tilde{K} \cdot{v_{{%(i)d}}}$" % {'i': i+1},
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=15)
