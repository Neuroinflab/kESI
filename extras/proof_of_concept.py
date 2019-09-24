import numpy as np
import matplotlib.pyplot as plt
import kesi
from kcsd import KCSD2D, utility_functions, csd_profile
from scipy import integrate, interpolate
from scipy.spatial import distance
import matplotlib.cm as cm
from datetime import datetime

#evaluate_spline = interpolate._bspl.evaluate_spline




def make_plot(xx, yy, zz, title, cmap=cm.bwr):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(zz))
    if t_max == 0:
        t_max = np.finfo(zz.dtype).eps
    levels = np.linspace(-1 * t_max, t_max, 32)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2g', ticks=ticks)
    return ax

def integrate_2d(csd_at, true_csd, ele_pos, h, csd_lims):
    csd_x, csd_y = csd_at
    xlin = csd_lims[0]
    ylin = csd_lims[1]
    Ny = ylin.shape[0]
    m = np.sqrt((ele_pos[0] - csd_x)**2 + (ele_pos[1] - csd_y)**2)
    m[m < 0.0000001] = 0.0000001
    y = np.arcsinh(h / m) * true_csd
    integral_1D = np.zeros(Ny)
    for i in range(Ny):
        integral_1D[i] = integrate.simps(y[:, i], ylin)

    integral = integrate.simps(integral_1D, xlin)
    return integral

def forward_method(ele_pos, csd_at, true_csd):
    pots = np.zeros(ele_pos.shape[0])
    xlin = csd_at[0, :, 0]
    ylin = csd_at[1, 0, :]
    h = 50. # distance between the electrode plane and the CSD plane
    conductivity = 1.0 # S/m
    for ii in range(ele_pos.shape[0]):
        pots[ii] = integrate_2d(csd_at, true_csd,
        [ele_pos[ii][0], ele_pos[ii][1]], h,
        [xlin, ylin])
    return pots / (2 * np.pi * conductivity)


xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
n_src_init = 1000
R_init = 1.
ext_x = 0.0
ext_y = 0.0
h = 50. # distance between the electrode plane and the CSD plane
conductivity = 1.0 # S/m

def do_kcsd(ele_pos, pots):
    pots = pots.reshape((len(ele_pos), 1)) # first time point
    return KCSD2D(ele_pos, pots, h=h, sigma=conductivity,
                  xmin=xmin, xmax=xmax,
                  ymin=ymin, ymax=ymax,
                  n_src_init=n_src_init,
                  src_type='gauss',
                  R_init=R_init)


class SourceGauss2DBase(object):
    def __init__(self, x, y, R, h, conductivity):
        self.x = x
        self.y = y
        self.R = R
        self.h = h
        self.conductivity = conductivity

    def csd(self, XY):
        var = (self.R / 3.0)**2
        d = distance.cdist(XY, [[self.x, self.y]]).flatten()
        return np.exp(-0.5 * d ** 2 / var) / (
                np.sqrt(2 * np.pi) ** 2 * var)
        #return np.exp(-0.5 * np.sqrt(self._dist2(XY)) ** 2 / var) / (
        #        np.sqrt(2 * np.pi) ** 2 * var)
        ##return np.exp(-0.5 / var * self._dist2(XY)) / (2*np.pi*var)

    def _dist2(self, XY):
        # X, Y = np.array(XY).T
        # return (self.x - X)**2 + (self.y - Y)**2
        return ((XY - np.array([self.x, self.y])) ** 2).sum(axis=1)



class InterpolatingGeneratorOfSourceGauss2D(list):
    class InterpolatedSourceGauss2D(SourceGauss2DBase):
        def __init__(self, x, y, R, h, conductivity, potentialInterpolator):
            super(InterpolatingGeneratorOfSourceGauss2D.InterpolatedSourceGauss2D,
                  self).__init__(x, y, R, h, conductivity)

            self.potentialInterpolator = potentialInterpolator
            #c = potentialInterpolator.c
            #self._spline_c = c.reshape(c.shape[0], -1)

        def potential(self, XY):
            return self.potentialInterpolator(np.sqrt(self._dist2(XY)))

    def __init__(self, sources, electrodes, R, h, conductivity,
                 dist_table_density=20,
                 points=None):
        self.R = R
        self.h = h
        self.conductivity = conductivity

        maxDist = max(distance.cdist(sources, dst, 'euclidean').max()
                      for dst in [electrodes, points]
                      if dst is not None
                      ) + R
        dists = np.logspace(0., np.log10(maxDist + 1.), dist_table_density) - 1
        potentials = np.array(list(map(self.forward_model, dists)))
        # potentialInterpolator = interpolate.make_interp_spline(dists, potentials,
        #                                                        k=3)
        potentialInterpolator = interpolate.interp1d(dists, potentials,
                                                     kind='cubic')
        super(InterpolatingGeneratorOfSourceGauss2D,
              self).__init__(self.InterpolatedSourceGauss2D(
                               x, y, R, h, conductivity,
                               potentialInterpolator)
                             for x, y in sources)

    def forward_model(self, dist):
        R = self.R
        var = (R / 3.0)**2
        pot, err = integrate.dblquad(self.int_pot_2D,
                                     -R, R,
                                     lambda x: -R,
                                     lambda x: R,
                                     args=(dist, var, self.h))
        #return pot * (0.25 / (np.pi**2 * var * self._set_conductivities))
        return pot * (0.5 / (np.pi * self.conductivity))

    def int_pot_2D(self, xp, yp, x, var, h):
        y = np.sqrt((x-xp)**2 + yp**2)
        if y < 0.00001:
            y = 0.00001
        dist2 = xp**2 + yp**2
        #return np.arcsinh(h/y) * np.exp(-0.5 * dist2 / var)
        return np.arcsinh(h / y) * (np.exp(-0.5 * np.sqrt(dist2) ** 2 / var) / (
                np.sqrt(2 * np.pi) ** 2 * var))
        #return np.arcsinh(h/y) * np.exp(-0.5 * dist2 / var) / (2 * np.pi * var)



def do_kesi(approximator, pots, ele_pos):
    d = approximator('csd', 'potential',
                     dict(zip(map(tuple, ele_pos),
                              pots.flatten())))
    return np.array([d[k] for k in zip(csd_x.flatten(), csd_y.flatten())]).reshape(100, 100)


csd_at = np.mgrid[0.:1.:100j,
                  0.:1.:100j]
csd_x, csd_y = csd_at
CSD_PROFILE = csd_profile.gauss_2d_small
true_csd = CSD_PROFILE(csd_at, seed=15)
ele_x, ele_y = np.mgrid[0.05: 0.95: 10j,
0.05: 0.95: 10j]
ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T

pots = forward_method(ele_pos, csd_at, true_csd)


startKCSD = datetime.now()
k = do_kcsd(ele_pos, pots)
endKCSD = datetime.now()
est_csd = k.values('CSD')
finalKCSD = datetime.now()

src_x, src_y, R = utility_functions.distribute_srcs_2D(csd_x, csd_y, n_src_init, ext_x, ext_y, R_init)


preKESI = datetime.now()
sources = InterpolatingGeneratorOfSourceGauss2D(np.vstack((src_x.flatten(), src_y.flatten())).T,
                                                ele_pos,
                                                R, h, conductivity,
                                                points=list(zip(csd_x.flatten(),
                                                                csd_y.flatten())))
startKESI = datetime.now()
original = kesi.KernelFieldApproximator(sources,
                                        nodes={'potential': ele_pos},
                                        points={'csd': np.vstack((csd_x.flatten(), csd_y.flatten())).T})

endKESI = datetime.now()
kesi_csd = do_kesi(original, pots, ele_pos)
finalKESI = datetime.now()

print('     \tTotal time \tConstructor\tEstimation \tPreprocessing')
print('kCSD:\t{:10f} \t{:10f} \t{:10f}'.format(
                    *[x.total_seconds()
                      for x in [finalKCSD - startKCSD,
                                endKCSD - startKCSD,
                                finalKCSD - endKCSD]]))
print('kESI:\t{:10f} \t{:10f} \t{:10f}\t{:10f}'.format(
                    *[x.total_seconds()
                      for x in [finalKESI - preKESI,
                                endKESI - startKESI,
                                finalKESI - endKESI,
                                startKESI - preKESI]]))


def cv(original, measured, lambdas):
    POTS = original._measurement_vector('potential', measured)
    KERNEL = original._kernels['potential']
    n = KERNEL.shape[0]
    I = np.identity(n - 1)
    IDX_N = np.arange(n)
    errors = []
    for l in lambdas:
        errors.append(0.)
        for i, p in zip(IDX_N, POTS[:, 0]):
            IDX = IDX_N[IDX_N != i]
            K = KERNEL[np.ix_(IDX, IDX)]
            P = POTS[IDX, :]
            CK = KERNEL[np.ix_([i], IDX)]
            EST = np.dot(CK,
                         np.linalg.solve(K + l * I, P))
                         # np.dot(np.linalg.inv(K + l * I),
                         #        P))
            errors[-1] += (EST[0, 0] - p) ** 2

    return errors

measured = dict(zip(map(tuple, ele_pos),
                    pots.flatten()))
lambdas = [0] + list(np.logspace(-16, 0, 100))

startCvKESI = datetime.now()
err = cv(original, measured, lambdas)
endCvKESI = datetime.now()

idx_min = np.argmin(err)
plt.figure()
plt.title('Leave one out: SSE({:g}) = {:g}'.format(lambdas[idx_min], err[idx_min]))
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('SSE')
plt.axhline(err[0])
plt.plot(lambdas[1:], err[1:])
plt.tight_layout()

startCvKCSD = datetime.now()
k.cross_validate(lambdas=np.array(lambdas), Rs=np.array([R]))
endCvKCSD = datetime.now()
print('CV: kESI {:g}s,\tkCSD {:g}s'.format((endCvKESI - startCvKESI).total_seconds(),
                                           (endCvKCSD - startCvKCSD).total_seconds()))

make_plot(csd_x, csd_y, true_csd, 'True CSD')
make_plot(csd_x, csd_y, est_csd[:,:,0], 'kCSD CSD')
make_plot(csd_x, csd_y, kesi_csd, 'kESI CSD (0)')
diff = kesi_csd - est_csd[:, :, 0]
if (diff != 0).any():
    make_plot(csd_x, csd_y, diff, 'kESI - kCSD CSD (0)')
else:
    print('kESI and kCSD are compatible')
make_plot(csd_x, csd_y, do_kesi(original.copy(
    regularization_parameter=err[idx_min]), pots, ele_pos), 'kESI CSD (cv: {:g})'.format(err[idx_min]))
make_plot(csd_x, csd_y, k.values('CSD')[:,:,0], 'kCSD CSD (cv:{.lambd:g})'.format(k))
plt.show()
