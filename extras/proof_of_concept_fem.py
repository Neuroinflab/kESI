# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

import numpy as np
import gc
from dolfin import Constant, Mesh, MeshFunction, FunctionSpace, TestFunction, TrialFunction, Function, Measure, inner, grad, assemble, KrylovSolver
from dolfin import Expression, DirichletBC

sigma_B = 1. / 300.  # S / cm
sigma_brain = Constant(sigma_B)
sigma_scalp = Constant(sigma_B)
sigma_csf = Constant(5 * sigma_B)
sigma_skull = Constant(sigma_B / 20.)

whitemattervol = 32
graymattervol = 64
csfvol = 96
skullvol = 128
scalp = 160

BRAIN_R = 7.9
SCALP_R = 9.0
WHITE_R = 7.5
RAD_TOL = 0.01
NECK_ANGLE = -np.pi / 3
NECK_AT = BRAIN_R * np.sin(NECK_ANGLE)

ele_coords = []

for altitude in np.linspace(NECK_ANGLE, np.pi / 2, 16):
    for azimuth in np.linspace(0, 2 * np.pi, int(round(np.cos(altitude) * 36 / np.pi))+1,
                               endpoint=False):
        r = SCALP_R - RAD_TOL
        ele_coords.append(r * np.array([np.cos(altitude) * np.sin(azimuth),
                                        np.sin(altitude),
                                        np.cos(altitude) * np.cos(azimuth)]))

# ele_coords.append(np.array([0, 0, BRAIN_R]))
# ele_coords.append(np.array([np.nan] * 3))
ele_coords = np.transpose(ele_coords)

PATH = '_meshes/sphere_4_lowres'
mesh = Mesh(PATH + '.xml')
# lowres: 5.1s
subdomains = MeshFunction("size_t", mesh, PATH + '_physical_region.xml')
# lowres: 1.4s
boundaries = MeshFunction("size_t", mesh, PATH + '_facet_region.xml')
# lowres: 12s

DEGREE = 3
V = FunctionSpace(mesh, "CG", DEGREE)
# lowres: 42s (first time: 58s)
v = TestFunction(V)
# lowres << 1s
potential_trial = TrialFunction(V)
# lowres << 1s
potential = Function(V)
# lowres < 1s

dx = Measure("dx")(subdomain_data=subdomains)
# lowres << 1s
ds = Measure("ds")(subdomain_data=boundaries)
# lowres << 1s
a = inner(sigma_brain * grad(potential_trial), grad(v)) * dx(whitemattervol) + \
    inner(sigma_brain * grad(potential_trial), grad(v)) * dx(graymattervol) + \
    inner(sigma_scalp * grad(potential_trial), grad(v)) * dx(scalp) + \
    inner(sigma_csf * grad(potential_trial), grad(v)) * dx(csfvol) + \
    inner(sigma_skull * grad(potential_trial), grad(v)) * dx(skullvol)
# lowres < 1s
TERMS_WITH_UNKNOWN = assemble(a)
# lowres: 120s

# L = Constant(0)*v*dx
# known_terms = assemble(L)
#
# for x_pos, y_pos, z_pos, val in [(0., 0., 7.85, 1),
#                                  (0., 0., 7.75, -1),
#                                  ]:
#     point = Point(x_pos, y_pos, z_pos)
#     delta = PointSource(V, point, val)
#     delta.apply(known_terms)

# L = Expression('exp(-0.5 * ((x[0] - sourcex) * (x[0] - sourcex) + (x[1] - sourcey)*(x[1] - sourcey) + (x[2] - sourcez)*(x[2] - sourcez))/sigma2)',
#                sourcex=0.,
#                sourcey=0.,
#                sourcez=6.,
#                sigma2=0.01,
#                degree=DEGREE)*v*dx
# known_terms = assemble(L)


csd = Expression(f'''
                  x[0]*x[0] + x[1]*x[1] + x[2]*x[2] <= {BRAIN_R**2} && x[1] > {NECK_AT}?
                  A * exp(-0.5 * (x[0]*x[0] + (x[1] - source_y)*(x[1] - source_y) + (x[2] - source_z)*(x[2] - source_z))/sigma_2):
                  0
                  ''',
                  source_z=7.85,
                  source_y=0,
                  sigma_2=0.1,
                  A=(2 * np.pi * 0.1) ** -1.5,
                  degree=DEGREE)
#B_inv = assemble(csd * Measure('dx', mesh))

# csd_at = interpolate(csd, V)
# csd_at(0, 0, 7.2) < 0

def boundary(x, on_boundary):
    return x[1] <= NECK_AT

bc = DirichletBC(V, Constant(0.), boundary)

solver = KrylovSolver("cg", "ilu")
solver.parameters["maximum_iterations"] = 1100
solver.parameters["absolute_tolerance"] = 1E-8
#solver.parameters["monitor_convergence"] = True

SOURCES = []
DBG = {}

TMP_FILENAME = f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}_.npz'
try:
    fh = np.load(TMP_FILENAME)

except FileNotFoundError:
    print('no previous results found')
    previously_solved = set()

else:
    SIGMA = fh['SIGMA']
    R = fh['R']
    ALTITUDE = fh['ALTITUDE']
    AZIMUTH = fh['AZIMUTH']
    POTENTIAL = fh['POTENTIAL']

    for s, r, al, az, pot in zip(SIGMA, R, ALTITUDE, AZIMUTH, POTENTIAL):
        row = [s, r, al, az]
        row.extend(pot)
        SOURCES.append(row)

    previously_solved = set(zip(SIGMA, R, ALTITUDE))

def f():
    global SOURCES, DBG, previously_solved

    for sigma in np.logspace(1, -0.5, 4):
        csd.sigma_2 = sigma ** 2
        csd.A = (2 * np.pi * sigma ** 2) ** -1.5
        for z in np.linspace(WHITE_R, BRAIN_R, int(round((BRAIN_R - WHITE_R) / sigma)) + 1):
            for altitude in np.linspace(0, np.pi/2, int(round(np.pi/2 * z / sigma)) + 2):
                if (sigma, z, altitude) in previously_solved:
                    print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\tALREADY SOLVED: SKIPPING')
                    continue

                csd.source_z = z * np.cos(altitude)
                csd.source_y = z * np.sin(altitude)

                A_inv = assemble(csd * Measure('dx', mesh))
                if A_inv <= 0:
                    print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\tFAILED MISERABLY ({A_inv:g} <= 0)')
                    DBG[sigma, z, altitude] = ('FAILED MISERABLY', A_inv)
                    continue

                L = csd * v * dx
                # lowres: 1.1s (first time: 3.3s)
                known_terms = assemble(L)
                # lowres: 8.3s (first time: 9.5s)
                terms_with_unknown = TERMS_WITH_UNKNOWN.copy()
                bc.apply(terms_with_unknown, known_terms)

                gc.collect()
                print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\t({A_inv:g})')
                try:
                    solver.solve(terms_with_unknown, potential.vector(), known_terms)
                    # lowres: 1300 s
                    # : 4900 s
                except RuntimeError as e:
                    print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\tFAILED ({A_inv:g})')
                    DBG[sigma, z, altitude] = ('FAILED', A_inv)
                    continue

                # ELE_ALT = np.dot([[1, 0, 0],
                #                   [0, np.cos(-altitude), -np.sin(-altitude)],
                #                   [0, np.sin(-altitude), np.cos(-altitude)]],
                #                  ele_coords)
                ELE_ALT = np.array(ele_coords)
                DBG[sigma, z, altitude] = ('SUCCEEDED', A_inv)
                for azimuth in np.linspace(0, 2*np.pi, int(round(2 * np.pi * np.cos(altitude) * z / sigma)) + 2, endpoint=False):
                    #print(f'{sigma}\t{z}\t{altitude}\t{azimuth}')
                    ELECTRODES = np.dot([[np.cos(-azimuth), 0, np.sin(-azimuth)],
                                         [0, 1, 0],
                                         [-np.sin(-azimuth), 0, np.cos(-azimuth)]],
                                        ELE_ALT)
                    # ELECTRODES[:, -1] = [0, 0, z]
                    row = [sigma, z, altitude, azimuth]
                    row.extend(potential(*loc) for loc in ELECTRODES.T)
                    SOURCES.append(row)

                SRC = np.array(SOURCES)
                np.savez_compressed(f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}_.npz',
                                    SIGMA=SRC[:, 0],
                                    R=SRC[:, 1],
                                    ALTITUDE=SRC[:, 2],
                                    AZIMUTH=SRC[:, 3],
                                    POTENTIAL=SRC[:, 4:],
                                    ELECTRODES=ele_coords)
                gc.collect()

    SRC = np.array(SOURCES)
    np.savez_compressed(f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}.npz',
                        SIGMA=SRC[:, 0],
                        R=SRC[:, 1],
                        ALTITUDE=SRC[:, 2],
                        AZIMUTH=SRC[:, 3],
                        POTENTIAL=SRC[:,4:],
                        ELECTRODES=ele_coords)

%time f()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
import colorblind_friendly as cbf

fh = np.load('proof_of_concept_fem_dirchlet_newman_CTX_rev2.npz')
ELECTRODES = fh['ELECTRODES']
ELECTRODES = pd.DataFrame(ELECTRODES.T, columns=['X', 'Y', 'Z'], index=[f'E{i + 1:03d}' for i in range(ELECTRODES.shape[1])])
POTENTIAL = pd.DataFrame(fh['POTENTIAL'], columns=ELECTRODES.index)
for k in ['SIGMA', 'R', 'ALTITUDE', 'AZIMUTH',]:
    POTENTIAL[k] = fh[k]

TOP_ELECTRODE = ELECTRODES.Y.idxmax()
E = ELECTRODES.loc[TOP_ELECTRODE]

for (r, sigma), DF in POTENTIAL[POTENTIAL.AZIMUTH == 0].groupby(['R', 'SIGMA']):
    ALTITUDE_DEG = DF.ALTITUDE * 180 / np.pi
    DISTANCE_MM = 10 * np.sqrt((DF.R * np.sin(DF.ALTITUDE) - E.Y) ** 2 + (
                                DF.R * np.cos(DF.ALTITUDE) - E.Z) ** 2 + E.X ** 2)

    fig, ax1 = plt.subplots(ncols=1, nrows=1, squeeze=True)
    ax1.set_title(f'R = {r * 10:.1f} mm; $\\sigma$ = {sigma * 10:.1f} mm')
    ax1.set_xlim(0, 90)
    ax1.set_xlabel('source altitude [deg]')
    ax1.set_ylabel('V', color=cbf.BLUE)
    ax2 = ax1.twinx()
    ax2.set_ylabel('distance [mm]', color=cbf.VERMILION)

    ax1.plot(ALTITUDE_DEG, DF[TOP_ELECTRODE],
             color=cbf.BLUE,
             ls='-',
             marker='|')
    ax2.plot(ALTITUDE_DEG, DISTANCE_MM,
             color=cbf.VERMILION,
             ls=':')

    fig.tight_layout()

plt.show()


from matplotlib.animation import FuncAnimation
import itertools

def palindrome(seqs):
    return list(itertools.chain(*(itertools.chain(seq, reversed(seq)) for seq in seqs)))


DF = POTENTIAL[(POTENTIAL.R == POTENTIAL.R.max()) & (POTENTIAL.SIGMA == POTENTIAL.SIGMA.min())]

def make_frames(DF, bys=[], palindrome=set()):
    if bys == []:
        return [ROW for _, ROW in DF.iterrows()]

    by = bys[0]
    bys_next = bys[1:]
    _by = [make_frames(TMP, bys_next, palindrome)
           for _, TMP in DF.groupby(by, sort=True)]
    if by in palindrome:
        _by.extend(reversed(_by))

    return sum(_by, [])

frames = make_frames(POTENTIAL, ['SIGMA', 'R', 'ALTITUDE', 'AZIMUTH'],
                     palindrome={'SIGMA', 'R', 'ALTITUDE'})

R = np.sqrt(ELECTRODES.X ** 2 + ELECTRODES.Z ** 2)
ALTITUDE = np.pi/2 - np.arctan2(ELECTRODES.Y, R)
X = ALTITUDE * ELECTRODES.X / R
Y = ALTITUDE * ELECTRODES.Z / R
r = max(np.abs(X).max(), np.abs(Y).max())
V = DF[[f'E{i+1:03d}' for i in range(len(ELECTRODES))]].max()
t_max = np.abs(V).max()
ticks = np.linspace(-t_max, t_max, 3, endpoint=True)

fig, ax = plt.subplots()
scatterplot = ax.scatter(X, Y, c=V, cmap=cm.PRGn, vmin=-t_max, vmax=t_max, s=100)
fig.colorbar(scatterplot,
             ax=ax,
             ticks=ticks,
             format='%.2g')

def init():
    ax.set_aspect("equal")
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    return scatterplot,

def update(ROW):
    V = ROW[[f'E{i+1:03d}' for i in range(len(ELECTRODES))]]
    ax.set_title(f' $\\sigma$ = {round(ROW.SIGMA * 10):g} mm, depth = {round((SCALP_R - ROW.R) * 10):g} mm, altitude = {ROW.ALTITUDE / np.pi * 180:.1f}, azimuth = {ROW.AZIMUTH / np.pi * 180:.1f}')
    scatterplot.set_color(scatterplot.cmap([scatterplot.norm(x) for x in V]))
    return scatterplot,

ani = FuncAnimation(fig, update, frames=frames[::10],
                    init_func=init,
                    #blit=True,
                    interval=1)

plt.show()


def f()
for i, (_, ROW) in enumerate(DF.iterrows()):
    if i % 10: continue


    V = ROW[[f'E{i+1:03d}' for i in range(len(ELECTRODES))]]
    R = np.sqrt(ELECTRODES.X ** 2 + ELECTRODES.Z ** 2)
    ALTITUDE = np.pi/2 - np.arctan2(ELECTRODES.Y, R)
    X = ALTITUDE * ELECTRODES.X / R
    Y = ALTITUDE * ELECTRODES.Z / R

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    r = max(np.abs(X).max(), np.abs(Y).max())
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f' $\\sigma$ = {round(ROW.SIGMA * 10):g} mm, depth = {round((SCALP_R - ROW.R) * 10):g} mm, altitude = {ROW.ALTITUDE / np.pi * 180:.1f}, azimuth = {ROW.AZIMUTH / np.pi * 180:.1f}')

    t_max = np.abs(V).max()
    ticks = np.linspace(-t_max, t_max, 3, endpoint=True)
    scatterplot = ax.scatter(X, Y, c=V, cmap=cm.bwr, vmin=-t_max, vmax=t_max, s=20)
    fig.colorbar(scatterplot,
                 ax=ax,
                 ticks=ticks,
                 format='%.2g')
