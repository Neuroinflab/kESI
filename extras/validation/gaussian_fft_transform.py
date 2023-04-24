"""
@author: mbejtka
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from common import GaussianSourceKCSD3D


H = 1e-2
conductivity = 0.3
X = np.linspace(0, 1, 101)
nr = 8
data = pd.read_csv('eigenvalues_by_m_sd_extremely_small_width.csv')
sigmas = list(data.sigma.values[:nr])

eigenvalues = data.iloc[:nr, 3:].values
sigma = H
for indx, sigma in enumerate(sigmas):
    sources = GaussianSourceKCSD3D(0.5, 0, 0, sigma, conductivity)
    gaussian = sources.csd(X, 0, 0)
    fft_gaussian = np.fft.rfft(gaussian)

    plt.figure(figsize=(10, 7))
    plt.suptitle('sigma = ' + str(sigma))
    plt.subplot(1, 3, 1)
    plt.plot(eigenvalues[indx, :])
    plt.xlabel('nr of components')
    plt.title('eigenvalues')
    plt.yscale('log')
    plt.ylim([1e-5, 3e5])
    plt.subplot(1, 3, 2)
    plt.plot(gaussian)
    plt.xlabel('nr od probes')
    #plt.title('Gauss')
    plt.subplot(1, 3, 3)
    plt.stem(abs(fft_gaussian))
    plt.xlabel('nr od probes')
    plt.title('FFT')
    plt.tight_layout()
    plt.show()
