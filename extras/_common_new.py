import logging
import collections

import numpy as np
from scipy.special import erf, lpmv


logger = logging.getLogger(__name__)


class GaussianSourceBase(object):
    def __init__(self, x, y, z, standard_deviation):
        self.x = x
        self.y = y
        self.z = z
        self._variance = standard_deviation ** 2
        self._a = (2 * np.pi * self._variance) ** -1.5


class GaussianSourceKCSD3D(GaussianSourceBase):
    _dtype = np.sqrt(0.5).__class__
    _fraction_of_erf_to_x_limit_in_0 = _dtype(2 / np.sqrt(np.pi))
    _x = _dtype(1.)
    _half = _dtype(0.5)
    _last = 2.
    _err = 1.
    while 0 < _err < _last:
        _radius_of_erf_to_x_limit_applicability = _x
        _last = _err
        _x *= _half
        _err = _fraction_of_erf_to_x_limit_in_0 - erf(_x) / _x

    def __init__(self, x, y, z, standard_deviation, conductivity):
        super(GaussianSourceKCSD3D, self).__init__(x, y, z, standard_deviation)
        self.conductivity = conductivity
        self._b = 0.25 / (np.pi * conductivity)
        self._c = np.sqrt(0.5) / standard_deviation

    def csd(self, X, Y, Z):
        return self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._variance)

    def potential(self, X, Y, Z):
        R = np.sqrt((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)
        Rc = R * self._c
        return self._b * np.where(Rc >= self._radius_of_erf_to_x_limit_applicability,
                                  erf(Rc) / R,
                                  self._c * self._fraction_of_erf_to_x_limit_in_0)


class FourSphereModel(object):
    """
    Based on https://github.com/Neuroinflab/fourspheremodel
    by Chaitanya Chintaluri
    """
    Properies = collections.namedtuple('FourSpheres',
                                       ['brain',
                                        'csf',
                                        'skull',
                                        'scalp',
                                        ])

    I = 10.
    n = np.arange(1, 100)

    def __init__(self, conductivity, radius):
        self._set_radii(radius)
        self._set_conductivities(conductivity)

    def _set_radii(self, radius):
        self.radius = radius
        self.r12 = radius.brain / radius.csf
        self.r23 = radius.csf / radius.skull
        self.r34 = radius.skull / radius.scalp
        self.r21 = 1. / self.r12
        self.r32 = 1. / self.r23
        self.r43 = 1. / self.r34



    def _set_conductivities(self, conductivity):
        self.conductivity = conductivity
        self.s12 = conductivity.brain / conductivity.csf
        self.s23 = conductivity.csf / conductivity.skull
        self.s34 = conductivity.skull / conductivity.scalp

    def V(self, n):
        k = (n+1.) / n
        Factor = ((self.r34**n - self.r43**(n+1))
                  / (k * self.r34**n + self.r43**(n+1)))
        num = self.s34 / k - Factor
        den = self.s34 + Factor
        return num / den

    def Y(self, n):
        k = n / (n+1.)
        V_n = self.V(n)
        r23n = self.r23 ** n
        r32n1 = self.r32 ** (n + 1)
        Factor = ((r23n * k - V_n * r32n1)
                  / (r23n + V_n * r32n1))
        return (self.s23 * k - Factor) / (self.s23 + Factor)

    def Z(self, n):
        k = (n+1.) / n
        Y_n = self.Y(n)
        r12n = self.r12 ** n
        r21n1 = self.r21 ** (n + 1)
        return (r12n - k * Y_n * r21n1) / (r12n + Y_n * r21n1)

    def __call__(self, loc, P):
        return self._PointDipole(self, loc, P)

    class _PointDipole(object):
        def __init__(self, model, loc, P):
            self.model = model
            self.loc = loc
            self.P = np.reshape(P, (1, -1))
            self.dp_rad, self.dp_tan, r = self.decompose_dipole()
            self._set_dipole_r(r)

        def decompose_dipole(self):
            dist_dp = np.linalg.norm(self.loc)
            dp_rad = (np.dot(self.P, self.loc) / dist_dp) * (self.loc / dist_dp)
            dp_tan = self.P - dp_rad
            return dp_rad, dp_tan, dist_dp

        def _set_dipole_r(self, r):
            self.rz = r
            self.rz1 = r / self.model.radius.brain
            # self.r1z = 1. / self.rz1

        def __call__(self, X, Y, Z):
            # P = np.reshape(P, (1, -1))
            ELECTRODES = np.vstack([X, Y, Z]).T
            # dp_rad, dp_tan, r = self.decompose_dipole(P, dp_loc)
            # self._set_dipole_r(r)

            adjusted_theta = self.adjust_theta(self.loc, ELECTRODES)
            adjusted_phi_angle = self.adjust_phi_angle(self.dp_tan,
                                                       self.loc,
                                                       ELECTRODES)

            sign_rad = np.sign(np.dot(self.P, self.loc))
            mag_rad = sign_rad * np.linalg.norm(self.dp_rad)
            mag_tan = np.linalg.norm(self.dp_tan)  # sign_tan * np.linalg.norm(dp_tan)

            coef = self.H(self.model.n, self.model.radius.scalp)
            cos_theta = np.cos(adjusted_theta)

            # radial
            n_coef = self.model.n * coef
            rad_coef = np.insert(n_coef, 0, 0)
            Lprod = np.polynomial.legendre.Legendre(rad_coef)
            Lfactor_rad = Lprod(cos_theta)
            rad_phi = mag_rad * Lfactor_rad

            # #tangential
            Lfuncprod = [np.sum([C * lpmv(1, P_val, ct)
                                 for C, P_val in zip(coef, self.model.n)])
                         for ct in cos_theta]

            tan_phi = -1 * mag_tan * np.sin(adjusted_phi_angle) * np.array(Lfuncprod)
            return (rad_phi + tan_phi) / (4 * np.pi * self.model.conductivity.brain * (self.rz ** 2))

        def adjust_theta(self, dp_loc, ele_pos):
            ele_dist = np.linalg.norm(ele_pos, axis=1)
            dist_dp = np.linalg.norm(dp_loc)
            cos_theta = np.dot(ele_pos, dp_loc) / (ele_dist * dist_dp)
            cos_theta = np.nan_to_num(cos_theta)
            theta = np.arccos(cos_theta)
            return theta

        def adjust_phi_angle(self, p, dp_loc, ele_pos):
            r_ele = np.sqrt(np.sum(ele_pos ** 2, axis=1))

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

        def H(self, n, r_ele):
            if r_ele < self.radius.brain:
                T1 = ((r_ele / self.radius.brain)**n) * self.A1(n)
                T2 = ((self.rz / r_ele)**(n + 1))
            elif r_ele < self.radius.csf:
                T1 = ((r_ele / self.radius.csf)**n) * self.A2(n)
                T2 = ((self.radius.csf / r_ele)**(n + 1)) * self.B2(n)
            elif r_ele < self.radius.skull:
                T1 = ((r_ele / self.radius.skull)**n) * self.A3(n)
                T2 = ((self.radius.skull / r_ele)**(n + 1)) * self.B3(n)
            elif r_ele <= self.radius.scalp:
                T1 = ((r_ele / self.radius.scalp)**n) * self.A4(n)
                T2 = ((self.radius.scalp / r_ele)**(n + 1)) * self.B4(n)
            else:
                print("Invalid electrode position")
                return
            return T1 + T2

        def A1(self, n):
            Z_n = self.Z(n)
            k = (n + 1.) / n
            return self.rz1 ** (n + 1) * (Z_n + self.s12 * k) / (self.s12 - Z_n)

        def A2(self, n):
            return ((self.A1(n) + self.rz1 ** (n + 1))
                    / (self.Y(n) * self.r21 ** (n + 1) + self.r12 ** n))

        def A3(self, n):
            return ((self.A2(n) + self.B2(n))
                    / (self.r23 ** n + self.V(n) * self.r32 ** (n + 1)))

        def B2(self, n):
            return self.A2(n) * self.Y(n)

        def A4(self, n):
            k = (n+1.) / n
            return k * ((self.A3(n) + self.B3(n))
                        / (k * self.r34 ** n + self.r43 ** (n + 1)))

        def B3(self, n):
            return self.A3(n) * self.V(n)

        def B4(self, n):
            return self.A4(n) * n / (n + 1.)

        def __getattr__(self, name):
            return getattr(self.model, name)


if __name__ == '__main__':
    import common
    import pandas as pd

    BRAIN_CONDUCTIVITY = 1. / 300.  # S / cm
    CONDUCTIVITY = common.FourSphereModel.Properies(1.00 * BRAIN_CONDUCTIVITY,
                                                    5.00 * BRAIN_CONDUCTIVITY,
                                                    0.05 * BRAIN_CONDUCTIVITY,
                                                    1.00 * BRAIN_CONDUCTIVITY)
    RADIUS = common.FourSphereModel.Properies(7.9, 8.0, 8.5, 9.0)
    X, Y, Z = np.meshgrid(np.linspace(-8.9, 8.9, 10),
                          np.linspace(-8.9, 8.9, 10),
                          np.linspace(-8.9, 8.9, 10))
    ELECTRODES = pd.DataFrame({'X': X.flatten(),
                               'Y': Y.flatten(),
                               'Z': Z.flatten(),
                               })
    DF = ELECTRODES.copy()
    oldFourSM = common.FourSphereModel(CONDUCTIVITY,
                                       RADIUS,
                                       ELECTRODES)
    newFourSM = FourSphereModel(CONDUCTIVITY,
                                RADIUS)

    src_pos = np.array([-0.05, 7.835, -0.0353])
    snk_pos = np.array([0.00, 7.765, 0.0353])
    DF['OLD'] = oldFourSM.compute_phi(src_pos, snk_pos)
    P = oldFourSM.I * (src_pos - snk_pos)
    LOC = 0.5 * (src_pos + snk_pos)
    newDipoleFourSM = newFourSM(LOC, P)
    DF['NEW'] = newDipoleFourSM(ELECTRODES.X,
                                ELECTRODES.Y,
                                ELECTRODES.Z)
    assert np.abs((DF.OLD - DF.NEW) / DF.OLD).max() < 1e-10


def cv(reconstructor, measured, regularization_parameters):
    errors = []

    for regularization_parameter in regularization_parameters:
        logger.info('cv(): error estimation for regularization parameter: {:g}'.format(regularization_parameter))
        ERR = np.array(reconstructor.leave_one_out_errors(measured,
                                                          regularization_parameter))
        errors.append(np.sqrt((ERR**2).mean()))

    return errors
