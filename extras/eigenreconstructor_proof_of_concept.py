#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

import matplotlib.pyplot as plt
import numpy as np

from kesi._verbose import VerboseFFR, _Eigenreconstructor
from kesi.common import GaussianSourceKCSD3D

import cbf


CONDUCTIVITY = 0.33  # S / m
H = 3e-4  # m
RES = 16
SD = H / RES
NS = 4 * RES
DS = 0.5 * H / NS
X_SOURCE = np.linspace(DS, H - DS, NS)
Y_SOURCE = X_SOURCE

sources = [GaussianSourceKCSD3D(x, y, 0, SD, CONDUCTIVITY)
           for x in X_SOURCE
           for y in Y_SOURCE]

class MeasurementManager(VerboseFFR.MeasurementManagerBase):
    def __init__(self, X, Y, Z=None):
        self.X = X.flatten()
        self.Y = Y.flatten()
        self.Z = np.zeros_like(self.X) if Z is None else Z.flatten()
        self.number_of_measurements = len(self.X)

    def probe(self, field):
        return field.potential(self.X, self.Y, self.Z)

measurement_manager = MeasurementManager(*np.meshgrid(np.linspace(0, H, 5),
                                                      np.linspace(0, H, 5)))


reconstructor = VerboseFFR(sources, measurement_manager)
eigenreconstructor = _Eigenreconstructor(reconstructor)

fig, ax = plt.subplots()
ax.set_title('Eigenvalues')
ax.set_yscale('log')
ax.plot(eigenreconstructor.EIGENVALUES[::-1],
        color=cbf.BLUE)

ev_min = eigenreconstructor.EIGENVALUES.min()
ev_max = eigenreconstructor.EIGENVALUES.max()
REGULARIZATION_PARAMETERS = [0] + sorted(set(np.logspace(np.log10(ev_min) - 1,
                                                         np.log10(ev_max) + 1,
                                                         100)) | set(eigenreconstructor.EIGENVALUES))
NOISE = [0, 0.01, 0.05, 0.1, 0.2]

eigensources = [eigenreconstructor._wrap_kernel_solution(v)
                for v in eigenreconstructor.EIGENVECTORS.T[::-1]]

CSD_X, CSD_Y = np.meshgrid(np.linspace(0, H, 101),
                           np.linspace(0, H, 101))


def error(E, O):
    return np.sqrt(np.square(E - O).mean())


for i, field in enumerate(eigensources, 1):
    fig, axes = plt.subplots(nrows=len(NOISE),
                             figsize=(6.4, 2.4 * len(NOISE)))
    #fig.suptitle('Eigensource #{}'.format(i))

    CSD_GT = field.csd(CSD_X, CSD_Y, 0)
    V_GT = measurement_manager.probe(field)
    amplitude = abs(V_GT).max()

    last_mask = False
    for noise, ax in zip(NOISE, axes):
        V = V_GT + noise * np.random.normal(size=V_GT.shape, scale=amplitude)
        err_eig = []
        err_reg = []
        for r in REGULARIZATION_PARAMETERS:
            print(i, noise, r)
            mask = eigenreconstructor.EIGENVALUES >= r
            if (last_mask != mask).any():
                last_mask = mask
                eig = eigenreconstructor(V, mask)
                err_eig.append(error(CSD_GT,
                                     eig.csd(CSD_X, CSD_Y, 0)))
            else:
                err_eig.append(err_eig[-1])

            reg = reconstructor(V, r)
            err_reg.append(error(CSD_GT,
                                 reg.csd(CSD_X, CSD_Y, 0)))

        ax.set_title('Noise (SD) = {:.1f}%'.format(noise * 100))
        ax.set_ylabel('MSE')
        ax.set_xscale('symlog', linthreshx=10 ** int(np.log10(ev_min)))
        for v in eigenreconstructor.EIGENVALUES:
            ax.axvline(v, ls=':', color=cbf.BLACK)
        ax.plot(REGULARIZATION_PARAMETERS, err_eig,
                color=cbf.VERMILION,
                label='Eig')
        ax.plot(REGULARIZATION_PARAMETERS, err_reg,
                color=cbf.BLUE,
                label='Reg')
        ax.legend(loc='best')

    fig.tight_layout()

plt.show()
