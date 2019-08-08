import operator
import gc

import numpy as np
import pandas as pd
import kesi

from common import (FourSphereModel,
                    GaussianSourceFEM,
                    PolarGaussianSourceFEM,
                    ElectrodeAwarePolarGaussianSourceKCSD3D,
                    ElectrodeAwareCartesianGaussianSourceKCSD3D,
                    ElectrodeAware)

NY = 41
NZ = 81

BRAIN_CONDUCTIVITY = 1. / 300.  # S / cm
CONDUCTIVITY = FourSphereModel.Properies(1.00 * BRAIN_CONDUCTIVITY,
                                         5.00 * BRAIN_CONDUCTIVITY,
                                         0.05 * BRAIN_CONDUCTIVITY,
                                         1.00 * BRAIN_CONDUCTIVITY)
RADIUS = FourSphereModel.Properies(7.9, 8.0, 8.5, 9.0)

WHITE_R = 7.5
RAD_TOL = 0.01


dipoles = {'rad': {'src_pos': [0., 7.85, 0.],
                   'snk_pos': [0., 7.75, 0.],
                   },
           'tan': {'src_pos': [0., 7.8, -0.05],
                   'snk_pos': [0., 7.8, 0.05],
                   },
           'mix': {'src_pos': [0., 7.835, -0.0353],
                   'snk_pos': [0., 7.765, 0.0353],
                   },
           'original_rad': {'src_pos': [0., 0., 7.85],
                            'snk_pos': [0., 0., 7.75],
                   },
           'original_tan': {'src_pos': [0., -0.05, 7.8],
                            'snk_pos': [0., 0.05, 7.8],
                   },
           'original_mix': {'src_pos': [0., -0.0353, 7.835],
                            'snk_pos': [0., 0.0353, 7.765],
                   },
           }


YY, ZZ = np.meshgrid(np.linspace(0, 8, NY),
                     np.linspace(-8, 8, NZ))

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

SOURCES_GM = POTENTIAL[POTENTIAL.SIGMA <= 1].copy()
SOURCES_GM['CONDUCTIVITY'] = CONDUCTIVITY.brain

SX, SY, SZ = np.meshgrid(np.linspace(-8, 8, 11),
                         np.linspace(0, 8, 5),
                         np.linspace(-8, 8, 11),)
SOURCES_UNIFORM = pd.DataFrame({'X': SX.flatten(),
                                'Y': SY.flatten(),
                                'Z': SZ.flatten(),
                                'SIGMA': 1.0,
                                'CONDUCTIVITY': CONDUCTIVITY.brain,
                                })

EX, EY, EZ = np.meshgrid(np.linspace(-8, 8, 7),
                         np.linspace(0, 8, 4),
                         np.linspace(-8, 8, 7),)
UNIFORM_ELECTRODES = pd.DataFrame({'X': EX.flatten(),
                                   'Y': EY.flatten(),
                                   'Z': EZ.flatten(),
                                   },
                                   index=['E{:03d}'.format(i) for i in range(EX.size)])

SETUPS = [(ElectrodeAwareCartesianGaussianSourceKCSD3D,
           ELECTRODES.loc[RECORDING_ELECTRODES],
           SOURCES_UNIFORM),
          (ElectrodeAwareCartesianGaussianSourceKCSD3D,
           UNIFORM_ELECTRODES,
           SOURCES_UNIFORM),
          (ElectrodeAwarePolarGaussianSourceKCSD3D,
           ELECTRODES.loc[RECORDING_ELECTRODES],
           SOURCES_GM),
          (PolarGaussianSourceFEM,
           ELECTRODES.loc[RECORDING_ELECTRODES],
           SOURCES_GM),
          ]

reconstructors = []

DF = []
for setup, (cls, electrodes, sources) in enumerate(SETUPS):
    print(cls.__name__)

    reconstructor = kesi.FunctionalKernelFieldReconstructor(
        [cls(electrodes, ROW) if issubclass(cls, ElectrodeAware) else cls(ROW)
         for _, ROW in sources.iterrows()],
        'potential',
        electrodes.index)
    reconstructors.append(reconstructor)
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

    fourSM = FourSphereModel(CONDUCTIVITY,
                             RADIUS,
                             ELECTRODES if electrodes is not UNIFORM_ELECTRODES else UNIFORM_ELECTRODES)

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

for setup, (cls, _, __) in enumerate(SETUPS):
    for dtype in [np.float32, np.float64]:
        TMP = DF[(DF.dtype == dtype.__name__) & (DF.setup == setup) & ~ DF.name.apply(operator.methodcaller('startswith', 'random'))]
        for norm in ['max', 'L2', 'L1']:
            assert (TMP['left_err_' + norm] >= TMP['right_err_' + norm]).all(), '#{} ({.__name__}) {.__name__} {}'.format(setup, cls, dtype, norm)

# for dtype in [np.float32, np.float64]:
#     TMP = DF[(DF.dtype == dtype.__name__) & DF.name.apply(operator.methodcaller('startswith', 'random'))]
#     for norm in ['max', 'L2', 'L1']:
#         assert (TMP['left_err_' + norm] >= TMP['right_err_' + norm]).all(), '{.__name__} {}'.format(dtype, norm)
