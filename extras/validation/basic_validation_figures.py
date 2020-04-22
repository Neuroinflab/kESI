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
        t_max = np.max(np.abs(values))#1
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
    plt.savefig(path + str(fig_title) +'.png', dpi=300)
    return


def generate_figure(X, Y, Z, values, values_type, IDX, layer, title, save_path=sys.path[0]):
    VALUES = np.zeros(X.shape)  
    for v, i, j, k in zip(values, *np.where(IDX)):  # place values in regular grid
        VALUES[i, j, k] = v
    make_plot(VALUES, values_type, X, Y, Z, save_path, idx=layer, fig_title=title)


# Estimation points
r = 0.09
X, Y, Z = np.meshgrid(np.linspace(-r, r, 30),
                      np.linspace(-r, r, 30),
                      np.linspace(-r, r, 30))
IDX = X**2 + Y**2 + Z**2 <=r**2
EST_X = X[IDX]
EST_Y = Y[IDX]
EST_Z = Z[IDX]

src_nr = 0
layer = 15
path = os.path.join('/home/marta/FEM/kESI/extras/validation/')

#### one sphere #####
sphere = 'one sphere'
#sphere = 'four spheres'
# True csd
true_csd = np.load(path + 'true_csd_0_one_sphere.npy')
generate_figure(X, Y, Z, true_csd, 'csd', IDX, layer, str(sphere)+' True csd')

# Est csd without regularization
est_csd = np.load(path + 'one_sphere_est_csd_0_1000_deg_1.npy')
generate_figure(X, Y, Z, est_csd, 'csd', IDX, layer, str(sphere)+' Est csd rp=0')

# Est csd with regularization
est_csd_r = np.load(path + 'one_sphere_est_csd_0_1000_deg_1_rp_0054.npy')
generate_figure(X, Y, Z, est_csd_r, 'csd', IDX, layer, str(sphere)+' Est csd rp=0054')

# Error of reconstruction without regularization
error = calculate_point_error(true_csd, est_csd)
generate_figure(X, Y, Z, error, 'err', IDX, layer, str(sphere)+' Error rp=0')

# Error of reconstruction with regularization
error_r = calculate_point_error(true_csd, est_csd_r)
generate_figure(X, Y, Z, error_r, 'err', IDX, layer, str(sphere)+' Error rp=0054')

## Reliability Map without regularization - wrong data, load all estimated data and calculate error from scratch
#with np.load(path + 'error_mean_1000.npz') as fh:
#    error_mean = fh['ERROR']
#generate_figure(X, Y, Z, error_mean, 'err', IDX, layer, str(sphere)+' Reliability map rp=0')
#
## Reliability Map with regularization
#with np.load(path + 'error_mean_1000_deg_1_rp_0054.npz') as fh:
#    error_mean_r = fh['ERROR_MEAN']
#generate_figure(X, Y, Z, error_r_mean, 'err', IDX, layer, str(sphere)+' Reliability map rp=0054')


#### four spheres #####
sphere = 'four spheres
# True csd
true_csd = np.load(path + 'four_spheres_true_csd_0_1000_deg_1_rp_0.npy')
generate_figure(X, Y, Z, true_csd, 'csd', IDX, layer, str(sphere)+' True csd')

# Est csd without regularization
est_csd = np.load(path + 'four_spheres_est_csd_0_1000_deg_1_rp_0.npy')
generate_figure(X, Y, Z, est_csd, 'csd', IDX, layer, str(sphere)+' Est csd rp=0')

# Est csd with regularization
est_csd_r = np.load(path + 'four_spheres_est_csd_0_1000_deg_1_rp_001.npy')
generate_figure(X, Y, Z, est_csd_r, 'csd', IDX, layer, str(sphere)+' Est csd rp=001')

# Error of reconstruction without regularization
error = calculate_point_error(true_csd, est_csd)
generate_figure(X, Y, Z, error, 'err', IDX, layer, str(sphere)+' Error rp=0')

# Error of reconstruction with regularization
error_r = calculate_point_error(true_csd, est_csd_r)
generate_figure(X, Y, Z, error_r, 'err', IDX, layer, str(sphere)+' Error rp=001')

## Reliability Map without regularization - wrong data, load all estimated data and calculate error from scratch
#error_mean = np.load(path + 'four_spheres_error_mean_1000_deg_1_rp_0.npy')
#generate_figure(X, Y, Z, error_mean, 'err', IDX, layer, str(sphere)+' Reliability map rp=0')
#
## Reliability Map with regularization
#error_mean_r = np.load(path + 'four_spheres_error_mean_1000_deg_1_rp_001.npy')
#generate_figure(X, Y, Z, error_r_mean, 'err', IDX, layer, str(sphere)+' Reliability map rp=001')

