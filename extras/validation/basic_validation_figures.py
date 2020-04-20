#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marta
"""

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import matplotlib.cm as cm
import os

def sigmoid_error(error):
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
    return sig_error


def make_plot(values, val_type, X, Y, Z, path, idx=15, fig_title=None):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.set_title(fig_title)
    if val_type == 'csd':
        cmap = cm.bwr
        t_max = np.max(np.abs(values))
        t_min = -t_max
    else:
        cmap = cm.Greys
        t_max = 1
        t_min = 0
    levels = np.linspace(t_min, t_max, 65)
    ax.set_aspect('equal')
    im = ax.contourf(X[idx, :, :], Z[idx, :, :], values[:, idx, :], levels=levels, cmap=cmap, alpha=1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_xticks([X.min(), 0, X.max()])
    ax.set_yticks([Z.min(), 0, Z.max()])
    ticks = np.linspace(t_min, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
    plt.savefig(path + str(fig_title) +'.png')
    return

path = os.path.join('/home/marta/FEM/kESI/extras/validation/')
# Estimation points
r = 0.09
X, Y, Z = np.meshgrid(np.linspace(-r, r, 30),
                                  np.linspace(-r, r, 30),
                                  np.linspace(-r, r, 30))
IDX = X**2 + Y**2 + Z**2 <=r**2
EST_X = X[IDX]
EST_Y = Y[IDX]
EST_Z = Z[IDX]

#### one sphere #####
# True csd
true_csd = np.load(path + 'true_csd_0_one_sphere.npy')
TRUE_CSD = np.zeros((30, 30, 30))
for v, i, j, k in zip(true_csd, *np.where(IDX)):
    TRUE_CSD[i, j, k] = v
title = 'True csd one sphere'
make_plot(TRUE_CSD, 'csd', X, Y, Z, path, idx=15, fig_title=title)

# Est csd without regularization
est_csd = np.load(path + 'one_sphere_est_csd_0_1000_deg_1.npy')
EST_CSD = np.zeros((30, 30, 30))
for v, i, j, k in zip(est_csd, *np.where(IDX)):
    EST_CSD[i, j, k] = v
title = 'Est csd one sphere rp=0'
make_plot(EST_CSD, 'csd', X, Y, Z, path, idx=15, fig_title=title)

# Est csd with regularization
est_csd_r = np.load(path + 'one_sphere_est_csd_0_1000_deg_1_rp_0054.npy')
EST_CSD_R = np.zeros((30, 30, 30))
for v, i, j, k in zip(est_csd_r, *np.where(IDX)):
    EST_CSD_R[i, j, k] = v
title = 'Est csd one sphere rp=0054'
make_plot(EST_CSD_R, 'csd', X, Y, Z, path, idx=15, fig_title=title)

# Error of reconstruction without regularization
error = np.load(path + 'one_sphere_error_0_1000_deg_1.npy')
error = sigmoid_error(error)
ERROR = np.zeros((30, 30, 30))
for v, i, j, k in zip(error, *np.where(IDX)):
    ERROR[i, j, k] = v
title = 'Error one sphere rp=0'
make_plot(ERROR, 'err', X, Y, Z, path, idx=15, fig_title=title)

# Error of reconstruction with regularization
error_r = np.load(path + 'one_sphere_error_0_1000_deg_1_rp_0054.npy')
error_r = sigmoid_error(error_r)
ERROR_R = np.zeros((30, 30, 30))
for v, i, j, k in zip(error_r, *np.where(IDX)):
    ERROR_R[i, j, k] = v
title = 'Error one sphere rp=0054'
make_plot(ERROR_R, 'err', X, Y, Z, path, idx=15, fig_title=title)

# Reliability Map without regularization
with np.load(path + 'error_mean_1000.npz') as fh:
    error_mean = fh['ERROR']
ERROR_MEAN = np.zeros((30, 30, 30))
for v, i, j, k in zip(error_mean, *np.where(IDX)):
    ERROR_MEAN[i, j, k] = v
title = 'Reliability map one sphere rp=0'
make_plot(ERROR_MEAN, 'err', X, Y, Z, path, idx=15, fig_title=title)

# Reliability Map with regularization
with np.load(path + 'error_mean_1000_deg_1_rp_0054.npz') as fh:
    error_mean_r = fh['ERROR_MEAN']
ERROR_MEAN_R = np.zeros((30, 30, 30))
for v, i, j, k in zip(error_mean_r, *np.where(IDX)):
    ERROR_MEAN_R[i, j, k] = v
title = 'Reliability map one sphere rp=0054'
make_plot(ERROR_MEAN_R, 'err', X, Y, Z, path, idx=15, fig_title=title)


#### four spheres #####
# True csd
true_csd = np.load(path + 'four_spheres_true_csd_0_1000_deg_1_rp_0.npy')
TRUE_CSD = np.zeros((30, 30, 30))
for v, i, j, k in zip(true_csd, *np.where(IDX)):
    TRUE_CSD[i, j, k] = v
title = 'True csd four spheres'
make_plot(TRUE_CSD, 'csd', X, Y, Z, path, idx=15, fig_title=title)

# Est csd without regularization
est_csd = np.load(path + 'four_spheres_est_csd_0_1000_deg_1_rp_0.npy')
EST_CSD = np.zeros((30, 30, 30))
for v, i, j, k in zip(est_csd, *np.where(IDX)):
    EST_CSD[i, j, k] = v
title = 'Est csd four spheres rp=0'
make_plot(EST_CSD, 'csd', X, Y, Z, path, idx=15, fig_title=title)

# Est csd with regularization
est_csd_r = np.load(path + 'four_spheres_est_csd_0_1000_deg_1_rp_001.npy')
EST_CSD_R = np.zeros((30, 30, 30))
for v, i, j, k in zip(est_csd_r, *np.where(IDX)):
    EST_CSD_R[i, j, k] = v
title = 'Est csd four spheres rp=001'
make_plot(EST_CSD_R, 'csd', X, Y, Z, path, idx=15, fig_title=title)

# Error of reconstruction without regularization
error = np.load(path + 'four_spheres_error_0_1000_deg_1_rp_0.npy')
error = sigmoid_error(error)
ERROR = np.zeros((30, 30, 30))
for v, i, j, k in zip(error, *np.where(IDX)):
    ERROR[i, j, k] = v
title = 'Error four spheres rp=0'
make_plot(ERROR, 'err', X, Y, Z, path, idx=15, fig_title=title)

# Error of reconstruction with regularization
error_r = np.load(path + 'four_spheres_error_0_1000_deg_1_rp_001.npy')
error_r = sigmoid_error(error_r)
ERROR_R = np.zeros((30, 30, 30))
for v, i, j, k in zip(error_r, *np.where(IDX)):
    ERROR_R[i, j, k] = v
title = 'Error four spheres rp=001'
make_plot(ERROR_R, 'err', X, Y, Z, path, idx=15, fig_title=title)

# Reliability Map without regularization
error_mean = np.load(path + 'four_spheres_error_mean_1000_deg_1_rp_0.npy')
ERROR_MEAN = np.zeros((30, 30, 30))
for v, i, j, k in zip(error_mean, *np.where(IDX)):
    ERROR_MEAN[i, j, k] = v
title = 'Reliability map four spheres rp=0'
make_plot(ERROR_MEAN, 'err', X, Y, Z, path, idx=15, fig_title=title)

# Reliability Map with regularization
error_mean_r = np.load(path + 'four_spheres_error_mean_1000_deg_1_rp_001.npy')
ERROR_MEAN_R = np.zeros((30, 30, 30))
for v, i, j, k in zip(error_mean_r, *np.where(IDX)):
    ERROR_MEAN_R[i, j, k] = v
title = 'Reliability map four spheres rp=001'
make_plot(ERROR_MEAN_R, 'err', X, Y, Z, path, idx=15, fig_title=title)

