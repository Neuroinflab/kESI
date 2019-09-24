import numpy as np
import pandas as pd
import gc
import logging

import kesi
from common import (FourSphereModel, PolarGaussianSourceFEM,
                    ElectrodeAwarePolarGaussianSourceKCSD3D,
                    ElectrodeAware, cv,
                    loadPotentialsAdHocNPZ)


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

logger = logging.getLogger(__name__)


class GroundedElectrodeAwarePolarGaussianSourceKCSD3D(ElectrodeAwarePolarGaussianSourceKCSD3D):
    def __init__(self, GND, ELECTRODES, ROW):
        super(GroundedElectrodeAwarePolarGaussianSourceKCSD3D, self).__init__(
            ELECTRODES, ROW)
        self._GND = GND

    def potential(self, electrodes):
        reference = super(GroundedElectrodeAwarePolarGaussianSourceKCSD3D,
                          self).potential(self._GND).mean()

        return super(GroundedElectrodeAwarePolarGaussianSourceKCSD3D,
                     self).potential(electrodes) - reference


dipoles = {'rad': {'src_pos': [0., 7.85, 0.],
                   'snk_pos': [0., 7.75, 0.],
                   },
           'tan': {'src_pos': [0., 7.8, -0.05],
                   'snk_pos': [0., 7.8, 0.05],
                   },
           'mix': {'src_pos': [0., 7.835, -0.0353],
                   'snk_pos': [0., 7.765, 0.0353],
                   },
           # 'original_rad': {'src_pos': [0., 0., 7.85],
           #                  'snk_pos': [0., 0., 7.75],
           #         },
           # 'original_tan': {'src_pos': [0., -0.05, 7.8],
           #                  'snk_pos': [0., 0.05, 7.8],
           #         },
           # 'original_mix': {'src_pos': [0., -0.0353, 7.835],
           #                  'snk_pos': [0., 0.0353, 7.765],
           #         },
           }

REGULARIZATION_PARAMETERS = np.logspace(-5, 5, 101)

YY, ZZ = np.meshgrid(np.linspace(-8, 8, NY),
                     np.linspace(-8, 8, NZ))


def decorate_CSD(ax, title, r):
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_ylim(-r, r)
    ax.set_xlim(-r, r)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Y (mm)')


def plot_CSD(fig, ax, cax, X, Y, CSD, t_max=None):
    if t_max is None:
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
    draw_dipole(ax, LOC, P)


def draw_dipole(ax, LOC, P, color=cbf.BLACK, linestyle='-'):
    x, y, z = 10 * (LOC - P)
    dx, dy, dz = 10 * 2 * P
    ax.arrow(z, y, dz, dy,
             color=color,
             linestyle=linestyle,
             width=0.2,
             head_width=2,
             head_length=4,
             length_includes_head=True)


def estimate_dipoles(loc, reconstructor, r, n):
    mesh = np.linspace(-r, r, 2 * n)
    X, Y, Z = np.meshgrid(mesh + loc[0],
                          mesh + loc[1],
                          mesh + loc[2])
    CSD = reconstructor.csd(X, Y, Z)
    X *= CSD
    Y *= CSD
    Z *= CSD
    return np.array([[A[n - i:n + i,
                        n - i:n + i,
                        n - i:n + i].mean() * (2 * i * r / n) ** 3 / 2
                      for A in [X, Y, Z]]
                     for i in range(1, n + 1)])

def estimate_dipole(LOC, reconstructor, r=0.5, n=10):
    X, Y, Z = LOC.reshape(-1, 1) + np.linspace(-r, r, n).reshape(1, -1)
    X, Y, Z = np.meshgrid(X, Y, Z)
    CSD = reconstructor.csd(X, Y, Z)
    RECONSTRUCTED = np.array([np.average(A, weights=CSD / (2 * r) ** 3)
                              for A in [X, Y, Z]])
    return 0.5 * (RECONSTRUCTED - LOC)

RESULTS = {}
for filename in [#'proof_of_concept_fem_dirchlet_newman_CTX_deg_1.npz',
                 'proof_of_concept_fem_dirchlet_newman_CTX_rev2.npz',
                 #'proof_of_concept_fem_dirchlet_newman_CTX_deg_3.npz',
                 ]:
    logger.info(f'loading {filename}...')
    POTENTIAL, ELECTRODES  = loadPotentialsAdHocNPZ(filename,
                                                    CONDUCTIVITY.brain)
    # fh = np.load(filename)
    # ELECTRODES = fh['ELECTRODES']
    # ELECTRODE_NAMES = [f'E{i + 1:03d}' for i in range(ELECTRODES.shape[1])]
    # ELECTRODES = pd.DataFrame(ELECTRODES.T, columns=['X', 'Y', 'Z'], index=ELECTRODE_NAMES)
    # POTENTIAL = pd.DataFrame(fh['POTENTIAL'], columns=ELECTRODES.index)
    # for k in ['SIGMA', 'R', 'ALTITUDE', 'AZIMUTH',]:
    #     POTENTIAL[k] = fh[k]
    #
    # POTENTIAL['CONDUCTIVITY'] = CONDUCTIVITY.brain

    fourSM = FourSphereModel(CONDUCTIVITY,
                             RADIUS,
                             ELECTRODES)

    GND_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] == 0).all()]
    RECORDING_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] != 0).any()]

    reconstructors = {name: kesi.FunctionalKernelFieldReconstructor(
                                    [cls(GND_ELECTRODES, ELECTRODES, ROW)
                                     if issubclass(cls, ElectrodeAware)
                                     else cls(ROW)
                                     for _, ROW in POTENTIAL[(POTENTIAL.SIGMA < 1) & (POTENTIAL.ALTITUDE > np.pi / 2 * 80 / 90)].iterrows()],
                                    'potential',
                                    RECORDING_ELECTRODES)
                     for name, cls in [('kCSD', GroundedElectrodeAwarePolarGaussianSourceKCSD3D),
                                       ('kESI', PolarGaussianSourceFEM),
                                       ]}
    gc.collect()

    PHI = pd.DataFrame(index=ELECTRODES.index)
    loc_dipole = {}
    for name, dipole in dipoles.items():
        logger.info('Computing potentials for dipole: ' + name)
        src_pos = dipole['src_pos']
        snk_pos = dipole['snk_pos']
        loc_dipole[name] = np.mean([src_pos, snk_pos], axis=0)


        S = pd.Series(fourSM.compute_phi(src_pos, snk_pos),
                      index=ELECTRODES.index)
        PHI[name] = S - S[GND_ELECTRODES].mean()

    for name in PHI.columns:
        V = PHI[name]

        results = {}
        RESULTS[filename, name] = results
        for method, reconstructor in reconstructors.items():
            ERR = cv(reconstructor, V, REGULARIZATION_PARAMETERS)
            regularization_parameter = REGULARIZATION_PARAMETERS[np.argmin(ERR)]
            approximator_cv = reconstructor(V, regularization_parameter)
            approximator = reconstructor(V)

            results[method, 'ERR'] = ERR
            results[method, 'regularization_parameter'] = regularization_parameter
            results[method, 'CSD_cv'] = approximator_cv.csd(0, YY, ZZ)
            results[method, 'CSD'] = approximator.csd(0, YY, ZZ)
            logger.info(f'{name}\t{method}\tCV')
            # n = 24 to improve cache performance
            results[method, 'dipole_cv'] = estimate_dipoles(loc_dipole[name],
                                                            approximator_cv,
                                                            n=24,
                                                            r=1.0)
            logger.info(f'{name}\t{method}\t')
            results[method, 'dipole'] = estimate_dipoles(loc_dipole[name],
                                                         approximator,
                                                         n=24,
                                                         r=1.0)

for (filename, name), results in RESULTS.items():
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f'{name} ({filename})')

    gs = gridspec.GridSpec(3, 10,
                           figure=fig,
                           width_ratios=[0.95, 0.05]*5,
                           hspace=0.8)


    R = np.sqrt(ELECTRODES.X ** 2 + ELECTRODES.Z ** 2)
    ALTITUDE = np.pi/2 - np.arctan2(ELECTRODES.Y, R)
    X = ALTITUDE * ELECTRODES.X / R
    Y = ALTITUDE * ELECTRODES.Z / R

    ax = fig.add_subplot(gs[1, 4])
    cax = fig.add_subplot(gs[1, 5])
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
        ax = fig.add_subplot(gs[1, 4*i+2:4*i+4])
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
            ax = fig.add_subplot(gs[2*j, 4*i+2])
            decorate_CSD(ax,
                         f'''CSD ({method}) $\\lambda = {
                         0 if reconstruction == "CSD"
                         else results[method, "regularization_parameter"]
                         :g}$''',
                         100)
            plot_CSD(fig, ax, fig.add_subplot(gs[2*j, 4*i+3]),
                     ZZ, YY, results[method, reconstruction],
                     t_max_csd)
            plot_dipole(ax,
                        dipoles[name])

            for r in [BRAIN_R,
                      # CSF_R,
                      # SKULL_R,
                      # SCALP_R,
                      ]:
                ax.add_patch(plt.Circle((0, 0),
                                        radius=10*r,
                                        ls=':',
                                        edgecolor='k',
                                        facecolor='none'))

            ax = fig.add_subplot(gs[2*j, 8*i:8*i+2])
            LOC = loc_dipole[name]
            P_R = results[method, 'dipole' + reconstruction[3:]]
            P_X = P_R[:, 2]
            P_Y = P_R[:, 1]

            ax.set_aspect('equal')
            ax.set_xlim(10 * (LOC[2] + min(-3., P_X.min())),
                        10 * (LOC[2] + max(3., P_X.max())))
            ax.set_ylim(10 * (LOC[1] + min(-3., P_Y.min())),
                        10 * (LOC[1] + max(3., P_Y.max())))
            plot_dipole(ax,
                        dipoles[name])
            plt.plot(10*(LOC[2] + P_R[:, 2]),
                     10*(LOC[1] + P_R[:, 1]),
                     color=cbf.YELLOW,
                     linestyle=':',
                     zorder=10)
            # draw_dipole(ax, loc_dipole[name], P_R,
            #             color=cbf.PURPLE,
            #             linestyle=':')

        DIFF = results['kCSD', reconstruction] - results['kESI', reconstruction]
        ax = fig.add_subplot(gs[2*j, 4])
        decorate_CSD(ax, 'CSD (kCSD - kESI)', 100)
        plot_CSD(fig, ax, fig.add_subplot(gs[2*j, 5]),
                 ZZ, YY, DIFF, t_max_csd)
        plot_dipole(ax, dipoles[name])

        for r in [BRAIN_R,
                  # CSF_R,
                  # SKULL_R,
                  # SCALP_R,
                  ]:
            ax.add_patch(plt.Circle((0, 0),
                                    radius=10 * r,
                                    ls=':',
                                    edgecolor='k',
                                    facecolor='none'))

plt.show()
