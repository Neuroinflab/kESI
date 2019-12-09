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


H = 1e-2
conductivity = 0.3
X, Y = np.mgrid[0.05: 0.95: 10j,
                0.05: 0.95: 10j]
ELECTRODES = pd.DataFrame({'X': X.flatten(),
                           'Y': Y.flatten(),
                           })

measurement_manager = MeasurementManager(ELECTRODES)
src_X, src_Y = np.mgrid[0.:1.:101j,
                        0.:1.:101j]
est_X, est_Y = np.mgrid[0.:1.:100j,
                        0.:1.:100j]
EST_POINTS = pd.DataFrame({'X': est_X.flatten(),
                           'Y': est_Y.flatten(),
                           })
measurement_manager_basis = MeasurementManager(EST_POINTS)

RESULTS = pd.DataFrame(columns=['sigma', 'M'] + ['eigenvalue_{:03d}'.format(i)
                       for i in range(100)])
sigma_min = H/1000
sigma_max = H
R = sigma_max/sigma_min
n = -1
while True:
    n += 1
    for k in range(1, 2**n):
        sigma = sigma_min*R**((2*k-1)/2**n)
        idx = len(RESULTS)
        RESULTS.loc[idx, :2] = [sigma, src_X.size]
        try:
            print(n, k, sigma, sep='\t', end='\t')
            sources = gaussian_source_factory_2d(src_X.flatten(),
                                                 src_Y.flatten(),
                                                 sigma,
                                                 conductivity)
            reconstructor = ValidateKESI(sources, measurement_manager)
            eigenvalues, _ = reconstructor._evd(reconstructor.kernel)
            RESULTS.loc[idx, 2:] = eigenvalues
        except Exception as e:
            print('failed', e, end='\t')
        else:
            print('success', end='\t')
        finally:
            RESULTS.to_csv('eigenvalues_by_m_sd_small_width.csv')
            print('saved')

#            reconstructor_list.append(reconstructor)
#            eigensources = reconstructor._eigensources(measurement_manager_basis)
#            eigensources_list.append(eigensources)


#plt.figure()
#for i in range(len(standard_deviation)):
#    plt.plot(reconstructor_list[i].eigenvalues, label=i)
#plt.yscale('log')
#plt.xlabel('Number of components')
#plt.ylabel('Eigenvalues')
#plt.legend()
#plt.show()
#
####
#fig = plt.figure(figsize=(18, 16))
#heights = [1, 1, 1, 1]
#
#gs = gridspec.GridSpec(4, 4, height_ratios=heights, hspace=0.6, wspace=0.5)
#nr_plts = 16
#
#for i in range(nr_plts):
#    ax = fig.add_subplot(gs[i], aspect='equal')
#
#    a = eigensources_list[7][:, i].reshape(len(est_X), len(est_X), 1)
#    cset = ax.contourf(est_X, est_Y, a[:, :, 0], cmap=cm.bwr)
#
#    ax.text(0.5, 1.05, r"$\tilde{K} \cdot{v_{{%(i)d}}}$" % {'i': i+1},
#            horizontalalignment='center', transform=ax.transAxes,
#            fontsize=15)
