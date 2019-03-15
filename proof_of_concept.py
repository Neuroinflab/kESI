import numpy as np
import matplotlib.pyplot as plt
import kesi
from kcsd import KCSD2D, utility_functions, csd_profile
from scipy import integrate, interpolate
import matplotlib.cm as cm
from datetime import datetime

#evaluate_spline = interpolate._bspl.evaluate_spline




def make_plot(xx, yy, zz, title, cmap=cm.bwr):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(zz))
    levels = np.linspace(-1 * t_max, t_max, 32)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
    return ax

def integrate_2d(csd_at, true_csd, ele_pos, h, csd_lims):
    csd_x, csd_y = csd_at
    xlin = csd_lims[0]
    ylin = csd_lims[1]
    Ny = ylin.shape[0]
    m = np.sqrt((ele_pos[0] - csd_x)**2 + (ele_pos[1] - csd_y)**2)
    m[m < 0.0000001] = 0.0000001
    y = np.arcsinh(2 * h / m) * true_csd
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
        var = (R / 3.0)**2
        return np.exp(-0.5 / var * self._dist2(XY)) / (2*np.pi*var)

    def _dist2(self, XY):
        # X, Y = np.array(XY).T
        # return (self.x - X)**2 + (self.y - Y)**2
        return ((XY - np.array([self.x, self.y])) ** 2).sum(axis=1)

#class SourceGauss2D(SourceGauss2DBase):
#    def potential(self, loc):
#        x, y = loc
#        dist = np.sqrt((self.x - x)**2 + (self.y - y)**2)
#        R = self.R
#        var = (R / 3.0)**2
#        pot, err = integrate.dblquad(self.int_pot_2D,
#                                     -R, R,
#                                     lambda x: -R,
#                                     lambda x: R,
#                                     args=(dist, var, self.h))
#        return pot / ((2.0*np.pi)**2 * var * self.conductivity)
#        # Potential basis functions bi_x_y
#
#    # ripped from KCSD
#    def int_pot_2D(self, xp, yp, x, var, h):
#        """FWD model function.
#        Returns contribution of a point xp,yp, belonging to a basis source
#        support centered at (0,0) to the potential measured at (x,0),
#        integrated over xp,yp gives the potential generated by a
#        basis source element centered at (0,0) at point (x,0)
#        Parameters
#        ----------
#        xp, yp : floats or np.arrays
#            point or set of points where function should be calculated
#        x :  float
#            position at which potential is being measured
#        R : float
#            The size of the basis function
#        h : float
#            thickness of slice
#        Returns
#        -------
#        pot : float
#        """
#        y = np.sqrt((x-xp)**2 + yp**2)
#        if y < 0.00001:
#            y = 0.00001
#        dist2 = xp**2 + yp**2
#        return np.arcsinh(h/y) * np.exp(-0.5 * dist2 / var)


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

    def __init__(self, sources, electrodes, R, h, conductivity, dist_table_density=20):
        self.R = R
        self.h = h
        self.conductivity = conductivity

        maxDist = np.sqrt(max((sx - ex)**2 + (sy - ey)**2
                          for sx, sy in sources
                          for ex, ey in electrodes))
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
        #return pot * (0.25 / (np.pi**2 * var * self.conductivity))
        return pot * (0.5 / (np.pi * self.conductivity))
        # Potential basis functions bi_x_y

    def int_pot_2D(self, xp, yp, x, var, h):
        y = np.sqrt((x-xp)**2 + yp**2)
        if y < 0.00001:
            y = 0.00001
        dist2 = xp**2 + yp**2
        #return np.arcsinh(h/y) * np.exp(-0.5 * dist2 / var)
        return np.arcsinh(h/y) * np.exp(-0.5 * dist2 / var) / (2 * np.pi * var)



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


# # Naive but exact; may take some time
# original = kesi.KernelFieldApproximator([SourceGauss2D(x, y, R, h, conductivity)
#                                          for x, y in zip(src_x.flatten(),
#                                                          src_y.flatten())
#                                          ]
#                                         nodes={'potential': list(zip(ele_x.flatten(), ele_y.flatten()))},
#                                         points={'csd': list(zip(csd_x.flatten(), csd_y.flatten()))})
preKESI = datetime.now()
sources = InterpolatingGeneratorOfSourceGauss2D(np.vstack((src_x.flatten(), src_y.flatten())).T,
                                                ele_pos,
                                                R, h, conductivity)
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

eps = np.finfo(float).eps * max(np.abs(x).max() for x in original._kernels.values())
make_plot(csd_x, csd_y, true_csd, 'True CSD')
make_plot(csd_x, csd_y, est_csd[:,:,0], 'kCSD CSD')
make_plot(csd_x, csd_y, kesi_csd, 'kESI CSD (0)')
make_plot(csd_x, csd_y, do_kesi(original.copy(lambda_=eps), pots, ele_pos), 'kESI CSD ({:g})'.format(eps))
make_plot(csd_x, csd_y, do_kesi(original.copy(lambda_=0.001), pots, ele_pos), 'kESI CSD (0.001)')
plt.show()
