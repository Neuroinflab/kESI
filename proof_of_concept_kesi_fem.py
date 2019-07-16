import numpy as np
import pandas as pd
import gc

import kesi
from common import (FourSphereModel, PolarGaussianSourceFEM,
                    PolarGaussianSourceKCSD3D, ElectrodeAware)


import matplotlib.pyplot as plt
from matplotlib import gridspec

import colorblind_friendly as cbf


NY = 41
NZ = 81

sigma_B = 1. / 300.  # S / cm
sigma_brain = sigma_B
sigma_scalp = sigma_B
sigma_csf = 5 * sigma_B
sigma_skull = sigma_B / 20.


BRAIN_R = 7.9
CSF_R = 8.0
SCALP_R = 9.0
SKULL_R = 8.5

BRAIN_CONDUCTIVITY = 1. / 300.  # S / cm
CONDUCTIVITY = FourSphereModel.Properies(1.00 * BRAIN_CONDUCTIVITY,
                                         5.00 * BRAIN_CONDUCTIVITY,
                                         0.05 * BRAIN_CONDUCTIVITY,
                                         1.00 * BRAIN_CONDUCTIVITY)
RADIUS = FourSphereModel.Properies(7.9, 8.0, 8.5, 9.0)


WHITE_R = 7.5
RAD_TOL = 0.01


class GroundedPolarGaussianSourceKCSD3D(PolarGaussianSourceKCSD3D):
    def __init__(self, ROW, ELECTRODES, GND):
        super(GroundedPolarGaussianSourceKCSD3D, self).__init__(ROW, ELECTRODES)
        self._GND = GND

    def potential(self, electrodes):
        reference = super(GroundedPolarGaussianSourceKCSD3D,
                          self).potential(self._GND).mean()

        return super(GroundedPolarGaussianSourceKCSD3D,
                     self).potential(electrodes) - reference


dipoles = {'rad': {'src_pos': [0., 7.85, 0.],
                   'snk_pos': [0., 7.75, 0.],
                   },
           'tan': {'src_pos': [0., 7.8, -0.05],
                   'snk_pos': [0., 7.8, 0.05],
                   },
           'mix': {'src_pos': [0., 7.835, -0.0353],
                   'snk_pos': [0., 7.764, 0.0353],
                   },
           'original_rad': {'src_pos': [0., 0., 7.85],
                            'snk_pos': [0., 0., 7.75],
                   },
           'original_tan': {'src_pos': [0., -0.05, 7.8],
                            'snk_pos': [0., 0.05, 7.8],
                   },
           'original_mix': {'src_pos': [0., -0.0353, 7.835],
                            'snk_pos': [0., 0.0353, 7.764],
                   },
           }

REGULARIZATION_PARAMETERS = np.logspace(-5, 5, 101)

YY, ZZ = np.meshgrid(np.linspace(0, 8, NY),
                     np.linspace(-8, 8, NZ))



def cv(reconstructor, measured, REGULARIZATION_PARAMETERS):
    POTS = reconstructor._measurement_vector(measured)
    KERNEL = reconstructor._kernel
    n = KERNEL.shape[0]
    I = np.identity(n - 1)
    IDX_N = np.arange(n)
    errors = []
    for regularization_parameter in REGULARIZATION_PARAMETERS:
        errors.append(0.)
        for i, p in zip(IDX_N, POTS[:, 0]):
            IDX = IDX_N[IDX_N != i]
            K = KERNEL[np.ix_(IDX, IDX)]
            P = POTS[IDX, :]
            CK = KERNEL[np.ix_([i], IDX)]
            EST = np.dot(CK,
                         np.linalg.solve(K + regularization_parameter * I, P))
            errors[-1] += (EST[0, 0] - p) ** 2

    return errors


def decorate_CSD(ax, title, r):
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_ylim(0, r)
    ax.set_xlim(-r, r)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Y (mm)')


def plot_CSD(fig, ax, cax, X, Y, CSD):
    t_max = np.abs(CSD).max()
    if t_max == 0:
        t_max = np.finfo(CSD.dtype).eps
    levels = np.linspace(-1 * t_max, t_max, 256)
    im = ax.contourf(X * 10, Y * 10, CSD,
                     levels=levels,
                     cmap=cbf.bwr)
    colorbar_ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    fig.colorbar(im,
                 cax=cax,
                 orientation='vertical',
                 format='%.2g',
                 ticks=colorbar_ticks)


def plot_dipole(ax, dipole):
    P, LOC = fourSM.get_dipole_moment_and_loc(fourSM.I,
                                              np.array(dipole['src_pos']),
                                              np.array(dipole['snk_pos']))
    x, y, z = 10 * (LOC - 0.5 * P)
    dx, dy, dz = 10 * P
    ax.arrow(z, y, dz, dy,
             color=cbf.BLACK,
             width=0.2,
             head_width=2,
             head_length=4,
             length_includes_head=True)




for filename in ['proof_of_concept_fem_dirchlet_newman_CTX_deg_1.npz',
                 'proof_of_concept_fem_dirchlet_newman_CTX_rev2.npz',
                 'proof_of_concept_fem_dirchlet_newman_CTX_deg_3.npz',
                 ]:
    print(f'loading {filename}...')
    fh = np.load(filename)
    ELECTRODES = fh['ELECTRODES']
    ELECTRODE_NAMES = [f'E{i + 1:03d}' for i in range(ELECTRODES.shape[1])]
    ELECTRODES = pd.DataFrame(ELECTRODES.T, columns=['X', 'Y', 'Z'], index=ELECTRODE_NAMES)
    POTENTIAL = pd.DataFrame(fh['POTENTIAL'], columns=ELECTRODES.index)
    for k in ['SIGMA', 'R', 'ALTITUDE', 'AZIMUTH',]:
        POTENTIAL[k] = fh[k]

    POTENTIAL['CONDUCTIVITY'] = CONDUCTIVITY.brain

    fourSM = FourSphereModel(CONDUCTIVITY,
                             RADIUS,
                             ELECTRODES)

    GND_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] == 0).all()]
    RECORDING_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] != 0).any()]

    reconstructors = {name: kesi.FunctionalKernelFieldReconstructor(
                                    [cls(ROW, ELECTRODES, GND_ELECTRODES)
                                     if issubclass(cls, ElectrodeAware)
                                     else cls(ROW)
                                     for _, ROW in POTENTIAL[POTENTIAL.SIGMA <= 1].iterrows()],
                                    'potential',
                                    RECORDING_ELECTRODES)
                     for name, cls in [('kCSD', GroundedPolarGaussianSourceKCSD3D),
                                       ('kESI', PolarGaussianSourceFEM),
                                       ]}
    gc.collect()

    PHI = pd.DataFrame(index=ELECTRODES.index)
    for name, dipole in dipoles.items():
        print('Now computing for dipole: ', name)
        src_pos = dipole['src_pos']
        snk_pos = dipole['snk_pos']

        S = pd.Series(fourSM.compute_phi(src_pos, snk_pos),
                      index=ELECTRODES.index)
        PHI[name] = S - S[GND_ELECTRODES].mean()

    for name in PHI.columns:
        V = PHI[name]

        results = {}
        for method, reconstructor in reconstructors.items():
            ERR = cv(reconstructor, V, REGULARIZATION_PARAMETERS)
            regularization_parameter = REGULARIZATION_PARAMETERS[np.argmin(ERR)]
            approximator_cv = reconstructor(V, regularization_parameter)
            approximator = reconstructor(V)

            results[method, 'ERR'] = ERR
            results[method, 'regularization_parameter'] = regularization_parameter
            results[method, 'CSD_cv'] = approximator_cv.csd(0, YY, ZZ)
            results[method, 'CSD'] = approximator.csd(0, YY, ZZ)

        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(f'{name} ({filename})')

        gs = gridspec.GridSpec(3, 6,
                               figure=fig,
                               width_ratios=[0.95, 0.05]*3,
                               hspace=0.4)


        R = np.sqrt(ELECTRODES.X ** 2 + ELECTRODES.Z ** 2)
        ALTITUDE = np.pi/2 - np.arctan2(ELECTRODES.Y, R)
        X = ALTITUDE * ELECTRODES.X / R
        Y = ALTITUDE * ELECTRODES.Z / R

        ax = fig.add_subplot(gs[1, 2])
        cax = fig.add_subplot(gs[1, 3])
        ax.set_title('Electric potential')
        ax.set_aspect("equal")
        r = max(np.abs(X).max(),
                np.abs(Y).max())
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_xticks([])
        ax.set_yticks([])
        t_max_v = np.abs(V).max()
        colorbar_ticks_v = np.linspace(-t_max_v, t_max_v, 3, endpoint=True)
        scatterplot = ax.scatter(X, Y, c=V, cmap=cbf.PRGn, vmin=-t_max_v, vmax=t_max_v, s=20)
        fig.colorbar(scatterplot,
                     cax=cax,
                     orientation='vertical',
                     ticks=colorbar_ticks_v,
                     format='%.2g')

        for i, method in enumerate(['kCSD', 'kESI']):
            ax = fig.add_subplot(gs[1, 4*i:4*i+2])
            ax.set_title(f'CV ({method})')
            ax.set_ylabel('SSE')
            ax.set_xlabel('$\\lambda$')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.plot(REGULARIZATION_PARAMETERS,
                    results[method, 'ERR'],
                    color=cbf.BLUE)
            ax.axvline(results[method, 'regularization_parameter'],
                       ls=':',
                       color=cbf.BLACK)

        for j, reconstruction in enumerate(['CSD', 'CSD_cv']):
            t_max_csd = max(np.abs(results[method, reconstruction]).max()
                            for method in ['kCSD', 'kESI'])

            for i, method in enumerate(['kCSD', 'kESI']):
                ax = fig.add_subplot(gs[2*j, 4*i])
                decorate_CSD(ax,
                             f'''CSD ({method}) $\\lambda = {
                             0 if reconstruction == "CSD"
                             else results[method, "regularization_parameter"]
                             :g}$''',
                             100)
                plot_CSD(fig, ax, fig.add_subplot(gs[2*j, 4*i+1]),
                         ZZ, YY, results[method, reconstruction],
                         t_max_csd)
                plot_dipole(ax,
                            dipoles[name])

            DIFF = results['kCSD', reconstruction] - results['kESI', reconstruction]
            ax = fig.add_subplot(gs[2*j, 2])
            decorate_CSD(ax, 'CSD (kCSD - kESI)', 100)
            plot_CSD(fig, ax, fig.add_subplot(gs[2*j, 3]),
                     ZZ, YY, DIFF, t_max_csd)



plt.show()
