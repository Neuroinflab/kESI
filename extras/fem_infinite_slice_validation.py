#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import cbf
from FEM import fem_finite_slice_gaussian as ffsg
import _common_new as common

logging.basicConfig(level=logging.INFO)

FIGURE_PATH = 'figures/0125'
NAMES = ['finite_slice_small_gaussian_0125_deg_2.npz0.npz',
         'finite_slice_small_gaussian_0125_deg_1.npz',
         'finite_slice_smaller_gaussian_0125_deg_2.npz',
         'finite_slice_smaller_gaussian_0125_deg_1.npz',
         'finite_slice_smaller_coarse_gaussian_0125_deg_2.npz',
         'finite_slice_smaller_coarse_gaussian_0125_deg_1.npz',
         ]

src_x = 0.
#src_y = 1.5e-4
src_z = 0.

factories = [ffsg.FiniteSliceGaussianSourceFactory(name) for name in NAMES]

factory = factories[1]

idx_x = np.argmin(abs(factory.X - np.array(src_x)))
# idx_y = np.argmin(abs(factory.X - np.array(src_y)))
idx_z = np.argmin(abs(factory.X - np.array(src_z)))
idx_xz = idx_x * (idx_x + 1) // 2 + idx_z

X, Y = np.meshgrid(np.linspace(factory.X_SAMPLE.min(),
                               factory.X_SAMPLE.max(),
                               5 * (len(factory.X_SAMPLE) - 1) + 1),
                   np.linspace(factory.Y_SAMPLE.min(),
                               factory.Y_SAMPLE.max(),
                               5 * (len(factory.Y_SAMPLE) - 1) + 1))
Z = np.zeros_like(X)

interactive = plt.isinteractive()
try:
    plt.interactive(False)
    for idx_y, src_y in enumerate(factory.X):

        ground_truth = common.InfiniteSliceSourceMOI(
            factory.X[idx_x],
            src_y,
            factory.X[idx_z],
            factory.slice_thickness,
            slice_conductivity=factory.slice_conductivity,
            saline_conductivity=factory.saline_conductivity,
            amplitude=(factory.a[idx_y, idx_xz]
                       / (2 * np.pi * factory.standard_deviation**2)**-1.5),
            standard_deviation=factory.standard_deviation)


        GT = ground_truth.potential(X, Y, Z)
        gt_amplitude = abs(GT).max()

        fems = [f(factory.X[idx_x],
                  src_y,
                  factory.X[idx_z])
                for f in factories]
        FEMS = [np.ma.masked_invalid(f.potential(X, Y, Z)) for f in fems]
        DIFFS = [(F - GT) / gt_amplitude for F in FEMS]
        v_max = np.max(np.ma.masked_invalid([abs(D).max() for D in DIFFS]))
        v_levels = np.linspace(-v_max, v_max, 256)

        fig = plt.figure(figsize=(2 * 5, len(NAMES) * 5))
        for i, (name, DIFF) in enumerate(zip(NAMES, DIFFS),
                                         start=1):
            ax = fig.add_subplot(len(NAMES), 1, i)
            ax.set_aspect('equal')
            ax.set_title('{} (ERR = {:.2f}%)'.format(name,
                                                    100 * abs(DIFF).max()))

            plot = ax.contourf(X, Y, DIFF,
                               levels=v_levels,
                               cmap=cbf.PRGn)
            fig.colorbar(plot, format='%.2e')

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_PATH,
                                 'DIFF_{:02d}.png'.format(idx_y)))
        plt.close(fig)

finally:
    plt.interactive(interactive)
