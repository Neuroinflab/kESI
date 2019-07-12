import operator
import gc

import numpy as np
import pandas as pd
import kesi

from scipy.special import lpmv, erf

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


YY, ZZ = np.meshgrid(np.linspace(0, 8, NY),
                     np.linspace(-8, 8, NZ))

class PolarBase(object):
    def __init__(self, ROW):
        y = ROW.R * np.sin(ROW.ALTITUDE)
        r = ROW.R * np.cos(ROW.ALTITUDE)
        x = r * np.sin(ROW.AZIMUTH)
        z = r * np.cos(ROW.AZIMUTH)
        self.init(x, y, z, ROW)


class CartesianBase(object):
    def __init__(self, ROW):
        self.init(ROW.x, ROW.y, ROW.z, ROW)


class GaussianSourceBase(object):
    def init(self, x, y, z, ROW):
        self.x = x
        self.y = y
        self.z = z
        self._sigma2 = ROW.SIGMA ** 2
        self._a = (2 * np.pi * self._sigma2) ** -1.5
        self._ROW = ROW

    def __getattr__(self, name):
        return getattr(self._ROW, name)


class GaussianSourceFEM(GaussianSourceBase):
    _BRAIN_R = 7.9
    NECK_ANGLE = -np.pi / 3
    NECK_AT = _BRAIN_R * np.sin(NECK_ANGLE)

    def csd(self, X, Y, Z):
        DIST2 = (X*X + Y*Y + Z*Z)
        return np.where((DIST2 <= self._BRAIN_R ** 2) & (Y > self.NECK_AT),
                        self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2),
                        0)

    def potential(self, electrodes):
        return self._ROW.loc[electrodes]


class GaussianSourceKCSD3D(GaussianSourceBase):
    CONDUCTIVITY = sigma_brain
    _b = 0.25 / (np.pi * CONDUCTIVITY)

    _dtype = np.sqrt(0.5).__class__
    _fraction_of_erf_to_x_limit_in_0 = _dtype(2 / np.sqrt(np.pi))
    _x = _dtype(1.)
    _half = _dtype(0.5)
    _last = 2.
    _err = 1.
    while _err < _last:
        _radius_of_erf_to_x_limit_applicability = _x
        _last = _err
        _x *= _half
        _err = np.abs(erf(_x) - _fraction_of_erf_to_x_limit_in_0)

    def init(self, x, y, z, ROW):
        super(GaussianSourceKCSD3D, self).init(x, y, z, ROW)
        self._c = np.sqrt(0.5) / ROW.SIGMA

    def csd(self, X, Y, Z):
        return self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2)

    def potential(self, electrodes):
        R = np.sqrt((electrodes.X - self.x) ** 2 + (electrodes.Y - self.y) ** 2 + (electrodes.Z - self.z) ** 2)
        Rc = R * self._c
        return self._b * np.where(Rc >= self._radius_of_erf_to_x_limit_applicability,
                                  erf(Rc) / R,
                                  self._c * self._fraction_of_erf_to_x_limit_in_0)


class PolarGaussianSourceFEM(PolarBase, GaussianSourceFEM):
    pass


class PolarGaussianSourceKCSD3D(PolarBase, GaussianSourceKCSD3D):
    pass


class CartesianGaussianSourceKCSD3D(CartesianBase, GaussianSourceKCSD3D):
    pass


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


filename = 'proof_of_concept_fem_dirchlet_newman_CTX_rev2.npz'
print(f'loading {filename}...')
fh = np.load(filename)
ELECTRODES = fh['ELECTRODES']
ELECTRODE_NAMES = [f'E{i + 1:03d}' for i in range(ELECTRODES.shape[1])]
ELECTRODES = pd.DataFrame(ELECTRODES.T, columns=['X', 'Y', 'Z'], index=ELECTRODE_NAMES)
POTENTIAL = pd.DataFrame(fh['POTENTIAL'], columns=ELECTRODES.index)
for k in ['SIGMA', 'R', 'ALTITUDE', 'AZIMUTH',]:
    POTENTIAL[k] = fh[k]



GND_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] == 0).all()]
RECORDING_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] != 0).any()]

SOURCES_GM = POTENTIAL[POTENTIAL.SIGMA <= 1]

SX, SY, SZ = np.meshgrid(np.linspace(-8, 8, 11),
                         np.linspace(0, 8, 5),
                         np.linspace(-8, 8, 11),)
SOURCES_UNIFORM = pd.DataFrame({'x': SX.flatten(),
                                'y': SY.flatten(),
                                'z': SZ.flatten(),
                                'SIGMA': 1.0,
                                })
EX, EY, EZ = np.meshgrid(np.linspace(-8, 8, 7),
                         np.linspace(0, 8, 4),
                         np.linspace(-8, 8, 7),)
UNIFORM_ELECTRODES = pd.DataFrame({'X': EX.flatten(),
                                   'Y': EY.flatten(),
                                   'Z': EZ.flatten(),
                                   })

CLS = [(CartesianGaussianSourceKCSD3D, ELECTRODES.loc[RECORDING_ELECTRODES], SOURCES_UNIFORM),
       (CartesianGaussianSourceKCSD3D, UNIFORM_ELECTRODES, SOURCES_UNIFORM),
       (PolarGaussianSourceKCSD3D, ELECTRODES.loc[RECORDING_ELECTRODES], SOURCES_GM),
       (PolarGaussianSourceFEM, RECORDING_ELECTRODES, SOURCES_GM),
       ]

DF = []
for setup, (cls, electrodes, sources) in enumerate(CLS):
    print(cls.__name__)

    reconstructor = kesi.FunctionalKernelFieldReconstructor(
        [cls(ROW) for _, ROW in sources.iterrows()],
        'potential',
        electrodes)
    gc.collect()

    A = np.array([f.csd(0, YY.flatten(), ZZ.flatten()) for f in reconstructor._field_components], dtype=np.float128).T
    B = reconstructor._pre_cross_kernel.astype(np.float128)

    N_ELS = reconstructor._pre_cross_kernel.shape[1]
    measures_names = ['electrode{:03d}'.format(i) for i in range(N_ELS)]
    measures = [np.identity(N_ELS)]
    if cls is GaussianSourceFEM:
        measures_names.append('tricky')
        measures.append([[0] * 18 + [26.640652943802397498] + [0] * 10 + [27.86522309371085808] + [0] * N_ELS - 30])

    measures_names.extend('noise{:03d}'.format(i) for i in range(100))
    np.random.seed(42)
    measures.append(np.random.normal(size=(100, N_ELS)))

    fourSM = FourSphereModel(ELECTRODES if electrodes is not UNIFORM_ELECTRODES else UNIFORM_ELECTRODES)
    fourSM.conductivity(sigma_skull)

    for name, dipole in dipoles.items():
        src_pos = dipole['src_pos']
        snk_pos = dipole['snk_pos']


        measures_names.append('dipole_' + name)
        if electrodes is not UNIFORM_ELECTRODES:
            S = pd.Series(fourSM.compute_phi(src_pos, snk_pos),
                          index=ELECTRODES.index)
            measures.append((S[RECORDING_ELECTRODES].values - S[GND_ELECTRODES].mean()).reshape(1, -1))
        else:
            measures.append(fourSM.compute_phi(src_pos, snk_pos))

    r_names = [(0, 'random{:03d}'.format(i)) for i in range(100)]
    C = [np.random.normal(size=(100, len(electrodes)))]

    for r in [0., 1., 10., 100.]:
        r_names.extend((r, name) for name in measures_names)
        C.append(reconstructor._solve_kernel(np.vstack(measures).T,
                                             regularization_parameter=r).T)

    C = np.vstack(C).T.astype(np.float128)

    A_BC = np.matmul(A, np.matmul(B, C))
    AB_C = np.matmul(np.matmul(A, B), C)
    ABC = 0.5 * (AB_C + A_BC)

    gc.collect()

    for dtype in [np.float128, np.float64, np.float32,]:
        print(dtype.__name__)
        if dtype is np.float128:
            # _AB_C, _A_BC = AB_C, A_BC
            _A, _B, _C = A, B, C
        else:
            _A, _B, _C = map(operator.methodcaller('astype', dtype),
                             [A, B, C])
            # _AB_C = np.matmul(np.matmul(_A, _B), _C)
            # _A_BC = np.matmul(_A, np.matmul(_B, _C))

        _AB = np.matmul(_A, _B)

        for i, (r, name) in enumerate(r_names):
            C_COL = _C[:, i]
            REF = ABC[:, i]
            L = np.matmul(_AB, C_COL) #_AB_C[:, i]
            R = np.matmul(_A, np.matmul(_B, C_COL))#_A_BC[:, i]
            row = {'regularization_parameter': r,
                   'setup': setup,
                   'sources': cls.__name__,
                   'name': name,
                   'dtype': dtype.__name__,
                   'diff_max': np.abs(L - R).max(),
                   'diff_L2': np.sqrt(((L - R)**2).mean()),
                   'diff_L1': np.abs(L - R).mean(),
                   'left_err_max': np.abs(REF - L).max(),
                   'right_err_max': np.abs(REF - R).max(),
                   'left_err_L2': np.sqrt(((REF - L)**2).mean()),
                   'right_err_L2': np.sqrt(((REF - R)**2).mean()),
                   'left_err_L1': np.abs(REF - L).mean(),
                   'right_err_L1': np.abs(REF - R).mean(),
                   }
            DF.append(row)

DF = pd.DataFrame(DF)

for setup, (cls, _, __) in enumerate(CLS):
    for dtype in [np.float32, np.float64]:
        TMP = DF[(DF.dtype == dtype.__name__) & (DF.setup == setup) & ~ DF.name.apply(operator.methodcaller('startswith', 'random'))]
        for norm in ['max', 'L2', 'L1']:
            assert (TMP['left_err_' + norm] >= TMP['right_err_' + norm]).all(), '#{} ({.__name__}) {.__name__} {}'.format(setup, cls, dtype, norm)

# for dtype in [np.float32, np.float64]:
#     TMP = DF[(DF.dtype == dtype.__name__) & DF.name.apply(operator.methodcaller('startswith', 'random'))]
#     for norm in ['max', 'L2', 'L1']:
#         assert (TMP['left_err_' + norm] >= TMP['right_err_' + norm]).all(), '{.__name__} {}'.format(dtype, norm)
