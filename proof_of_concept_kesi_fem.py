import numpy as np
import pandas as pd
import kesi
import gc

from scipy.special import lpmv

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
WHITE_R = 7.5
RAD_TOL = 0.01


dipoles = {'rad': {'src_pos': [0., 7.85, 0.],
                   'snk_pos': [0., 7.75, 0.],
                   },
           'tan': {'src_pos': [0., 7.8, -0.05],
                   'snk_pos': [0., 7.8, 0.05],
                   },
           'mix': {'src_pos': [0., 7.835, -0.0353],
                   'snk_pos': [0., 7.764, 0.0353],
                   },
           }

REGULARIZATION_PARAMETERS = np.logspace(-5, 5, 101)

YY, ZZ = np.meshgrid(np.linspace(0, 8, NY),
                     np.linspace(-8, 8, NZ))


class GaussianSurceFEM(object):
    _BRAIN_R = 7.9
    NECK_ANGLE = -np.pi / 3
    NECK_AT = _BRAIN_R * np.sin(NECK_ANGLE)

    def __init__(self, ROW):
        self._sigma2 = ROW.SIGMA ** 2
        self._a = (2 * np.pi * self._sigma2) ** -1.5
        self.y = ROW.R * np.sin(ROW.ALTITUDE)
        r = ROW.R * np.cos(ROW.ALTITUDE)
        self.x = r * np.sin(ROW.AZIMUTH)
        self.z = r * np.cos(ROW.AZIMUTH)
        self._ROW = ROW

    def csd(self, X, Y, Z):
        DIST2 = (X*X + Y*Y + Z*Z)
        return np.where((DIST2 <= self._BRAIN_R ** 2) & (Y > self.NECK_AT),
                        self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2),
                        0)

    def potential(self, electrodes):
        return self._ROW.loc[electrodes]





class FourSphereModel(object):
    """
    Based on https://github.com/Neuroinflab/fourspheremodel
    by Chaitanya Chintaluri
    """
    DIPOLE_R = 7.8
    rz = DIPOLE_R
    rz1 = rz / BRAIN_R
    r12 = BRAIN_R / CSF_R
    r23 = CSF_R / SKULL_R
    r34 = SKULL_R/ SCALP_R

    r1z = 1. / rz1
    r21 = 1. / r12
    r32 = 1. / r23
    r43 = 1. / r34

    I = 10.
    n = np.arange(1, 100)

    def __init__(self, ELECTRODES):
        self.ELECTRODES = ELECTRODES

    def V(self, n):
        k = (n+1.) / n
        Factor = ( (self.r34**n - (self.r43**(n+1)) ) / ( (k*(self.r34**n)) + (self.r43**(n+1)) ) )
        num = (self.s34/k) - Factor
        den = self.s34 + Factor
        return num / den


    def Y(self, n):
        k = n / (n+1.)
        Factor = ( ( (self.r23**n) * k) - self.V(n)*(self.r32**(n+1))) / (self.r23**n + self.V(n)*(self.r32**(n+1)))
        num = (self.s23*k) - Factor
        den = self.s23 + Factor
        return num / den


    def Z(self, n):
        k = (n+1.) / n
        num = (self.r12**n - k*self.Y(n)*(self.r21**(n+1)) ) / (self.r12**n + self.Y(n)*(self.r21**(n+1)))
        return num


    def A1(self, n):
        num = (self.rz1**(n+1))* (self.Z(n) + self.s12*((n+1.)/n))
        den = self.s12 - self.Z(n)
        return num / den


    def A2(self, n):
        num = self.A1(n) + (self.rz1**(n+1))
        den = (self.Y(n)*(self.r21**(n+1))) + self.r12**n
        return num / den


    def B2(self, n):
        return self.A2(n)*self.Y(n)


    def A3(self, n):
        num = self.A2(n) + self.B2(n)
        den = self.r23**n + (self.V(n)*(self.r32**(n+1)))
        return num / den


    def B3(self, n):
        return self.A3(n)*self.V(n)


    def A4(self, n):
        num = self.A3(n) + self.B3(n)
        k = (n+1.) / n
        den = (k*(self.r34**n)) + (self.r43**(n+1))
        return k*(num / den)


    def B4(self, n):
        return self.A4(n)* (n / (n+1.))


    def H(self, n, r_ele=SCALP_R):
        if r_ele < BRAIN_R:
            T1 = ((r_ele / BRAIN_R)**n) * self.A1(n)
            T2 = ((self.rz / r_ele)**(n + 1))
        elif r_ele < CSF_R:
            T1 = ((r_ele / CSF_R)**n) * self.A2(n)
            T2 = ((CSF_R / r_ele)**(n + 1)) * self.B2(n)
        elif r_ele < SKULL_R:
            T1 = ((r_ele / SKULL_R)**n) * self.A3(n)
            T2 = ((SKULL_R / r_ele)**(n + 1)) * self.B3(n)
        elif r_ele <= SCALP_R:
            T1 = ((r_ele / SCALP_R)**n) * self.A4(n)
            T2 = ((SCALP_R / r_ele)**(n + 1)) * self.B4(n)
        else:
            print("Invalid electrode position")
            return
        return T1 + T2


    def adjust_theta(self, src_pos, snk_pos):
        ele_pos = self.ELECTRODES.values
        dp_loc = (np.array(src_pos) + np.array(snk_pos)) / 2.
        ele_dist = np.linalg.norm(ele_pos, axis=1)
        dist_dp = np.linalg.norm(dp_loc)
        cos_theta = np.dot(ele_pos, dp_loc) / (ele_dist * dist_dp)
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        return theta


    def adjust_phi_angle(self, p, src_pos, snk_pos):
        ele_pos = self.ELECTRODES.values
        r_ele = np.sqrt(np.sum(ele_pos ** 2, axis=1))
        dp_loc = (np.array(src_pos) + np.array(snk_pos)) / 2.
        proj_rxyz_rz = (np.dot(ele_pos, dp_loc) / np.sum(dp_loc **2)).reshape(len(ele_pos),1) * dp_loc.reshape(1, 3)
        rxy = ele_pos - proj_rxyz_rz
        x = np.cross(p, dp_loc)
        cos_phi = np.dot(rxy, x.T) / np.dot(np.linalg.norm(rxy, axis=1).reshape(len(rxy),1), np.linalg.norm(x, axis=1).reshape(1, len(x)))
        cos_phi = np.nan_to_num(cos_phi)
        phi_temp = np.arccos(cos_phi)
        phi = phi_temp
        range_test = np.dot(rxy, p.T)
        for i in range(len(r_ele)):
            for j in range(len(p)):
                if range_test[i, j] < 0:
                    phi[i,j] = 2 * np.pi - phi_temp[i, j]
        return phi.flatten()


    def decompose_dipole(self, I, src_pos, snk_pos):
        P, dp_loc = self.get_dipole_moment_and_loc(I,
                                                   np.array(src_pos),
                                                   np.array(snk_pos))
        P = P.reshape((1, -1))

        dist_dp = np.linalg.norm(dp_loc)
        dp_rad = (np.dot(P, dp_loc) / dist_dp) * (dp_loc / dist_dp)
        dp_tan = P - dp_rad
        return P, dp_rad, dp_tan

    def get_dipole_moment_and_loc(self, I, SRC, SNK):
        return (I * (SRC - SNK)), (0.5 * (SRC + SNK))

    def conductivity(self, sigma_skull):
        self.s12 = sigma_brain /sigma_csf
        self.s23 = sigma_csf / sigma_skull
        self.s34 = sigma_skull / sigma_scalp

    def compute_phi(self, src_pos, snk_pos):
        P, dp_rad, dp_tan = self.decompose_dipole(self.I, src_pos, snk_pos)
        adjusted_theta = self.adjust_theta(src_pos, snk_pos)

        adjusted_phi_angle = self.adjust_phi_angle(dp_tan, src_pos, snk_pos)  # params.phi_angle_r

        dp_loc = (np.array(src_pos) + np.array(snk_pos)) / 2
        sign_rad = np.sign(np.dot(P, dp_loc))
        mag_rad = sign_rad * np.linalg.norm(dp_rad)
        mag_tan = np.linalg.norm(dp_tan)  # sign_tan * np.linalg.norm(dp_tan)

        coef = self.H(self.n)
        cos_theta = np.cos(adjusted_theta)

        # radial
        n_coef = self.n * coef
        rad_coef = np.insert(n_coef, 0, 0)
        Lprod = np.polynomial.legendre.Legendre(rad_coef)
        Lfactor_rad = Lprod(cos_theta)
        rad_phi = mag_rad * Lfactor_rad

        # #tangential
        Lfuncprod = []
        for tt in range(len(self.ELECTRODES)):
            Lfuncprod.append(np.sum([C * lpmv(1, P_val, cos_theta[tt])
                                     for C, P_val in zip(coef, self.n)]))
        tan_phi = -1 * mag_tan * np.sin(adjusted_phi_angle) * np.array(Lfuncprod)

        return (rad_phi + tan_phi) / (4 * np.pi * sigma_brain * (self.rz**2))



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

    sources = [GaussianSurceFEM(ROW) for _, ROW in POTENTIAL[POTENTIAL.SIGMA <= 1].iterrows()]

    fourSM = FourSphereModel(ELECTRODES)

    GND_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] == 0).all()]
    RECORDING_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] != 0).any()]

    reconstructor = kesi.FunctionalKernelFieldReconstructor(sources,
                                                            'potential',
                                                            RECORDING_ELECTRODES)
    gc.collect()

    phi = pd.DataFrame(index=ELECTRODES.index)
    for name, dipole in dipoles.items():
        print('Now computing for dipole: ', name)
        src_pos = dipole['src_pos']
        snk_pos = dipole['snk_pos']

        fourSM.conductivity(sigma_skull)
        S = pd.Series(fourSM.compute_phi(src_pos, snk_pos),
                      index=ELECTRODES.index)
        phi[name] = S - S[GND_ELECTRODES].mean()

    for name in phi.columns:
        V = phi[name]
        #V = ROW[[f'E{i+1:03d}' for i in range(len(ELECTRODES))]]
        CSD = reconstructor(V).csd(0, YY, ZZ)

        errors = cv(reconstructor, V, REGULARIZATION_PARAMETERS)
        regularization_parameter = REGULARIZATION_PARAMETERS[np.argmin(errors)]

        approximator_cv = reconstructor(V, regularization_parameter)
        CSD_cv = approximator_cv.csd(0, YY, ZZ)


        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(f'{name} ({filename})')

        gs = gridspec.GridSpec(1, 2, figure=fig)
        gsL = gridspec.GridSpecFromSubplotSpec(1, 2,
                                               subplot_spec=gs[:,0],
                                               width_ratios=[0.95, 0.05])
        gsR = gridspec.GridSpecFromSubplotSpec(3, 2,
                                               subplot_spec=gs[:,1],
                                               width_ratios=[0.95, 0.05])


        R = np.sqrt(ELECTRODES.X ** 2 + ELECTRODES.Z ** 2)
        ALTITUDE = np.pi/2 - np.arctan2(ELECTRODES.Y, R)
        X = ALTITUDE * ELECTRODES.X / R
        Y = ALTITUDE * ELECTRODES.Z / R

        ax = fig.add_subplot(gsL[0, 0])
        cax = fig.add_subplot(gsL[0, 1])
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


        ax = fig.add_subplot(gsR[0, 0])
        decorate_CSD(ax, 'CSD $\\lambda = 0$', 100)
        plot_CSD(fig, ax, fig.add_subplot(gsR[0, 1]),
                 ZZ, YY, CSD)
        plot_dipole(ax, dipoles[name])


        ax = fig.add_subplot(gsR[1, :])
        ax.set_title(f'CV')
        ax.set_ylabel('SSE')
        ax.set_xlabel('$\\lambda$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.plot(REGULARIZATION_PARAMETERS, errors, color=cbf.BLUE)
        ax.axvline(regularization_parameter, ls=':', color=cbf.BLACK)

        ax = fig.add_subplot(gsR[2, 0])
        decorate_CSD(ax, f'CSD $\\lambda = {regularization_parameter:g}$', 100)
        plot_CSD(fig, ax, fig.add_subplot(gsR[2, 1]),
                 ZZ, YY, CSD_cv)
        plot_dipole(ax, dipoles[name])

plt.show()
