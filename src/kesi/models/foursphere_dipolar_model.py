import warnings

import numpy as np
from scipy.special import lpmv


class PointDipole(object):
    def __init__(self, model, dipole_loc, dipole_moment):
        self.model = model
        self.set_dipole_loc(dipole_loc)
        self.decompose_dipole(dipole_moment)
        self._set_dipole_r()

    def set_dipole_loc(self, loc):
        self.loc_r = np.sqrt(np.square(loc).sum())

        if self.model.precision == 'float128':
            default_vector = np.array([[0, 0, 1]], dtype=np.float128)
        else:
            default_vector = np.array([[0, 0, 1]])

        self.loc_v = (loc / self.loc_r
                      if self.loc_r != 0
                      else default_vector)

    @property
    def loc(self):
        return self.loc_r * self.loc_v.flatten()

    @property
    def rz(self):
        return self.loc_r

    def decompose_dipole(self, P):
        self.p_rad = self.north_vector(P)
        self.p_tan = P - self.p_rad

    def north_vector(self, V):
        return np.matmul(self.north_projection(V),
                         self.loc_v)

    def north_projection(self, V):
        return np.matmul(V,
                         self.loc_v.T)

    def _set_dipole_r(self):
        self.rz1 = self.loc_r / self.model.radius.brain

    def __call__(self, X, Y, Z):
        """
        Samples the potential of point source in N-spheres, in Volts.


        Params:
          - X, Y, Z: three one dimensional arrays, of the same length,
                     together they define points in 3D space to sample the potential
        """
        if self.model.precision == 'float128':
            ELECTRODES = np.vstack([X, Y, Z], dtype=np.float128).T
        else:
            ELECTRODES = np.vstack([X, Y, Z]).T

        ele_dist = np.linalg.norm(ELECTRODES, axis=1)
        COS_THETA = self.cos_theta(ELECTRODES / ele_dist.reshape(-1, 1))
        tan_cosinus = self.tan_versor_cosinus(ELECTRODES).flatten()

        COEF = self.H_v(ele_dist)
        COEF_RAD = self.H_v(ele_dist, rad=True)
        if self.model.precision == 'float128':
            LPMV = lpmv(1,  # expensive for n >= 10_000;
                        self.n.reshape(1, -1),  # line_profiler claims 99.7%
                        COS_THETA.astype(np.float64)).astype(np.float128)  # experimental complexity O(n^2)
        else:
            LPMV = lpmv(1,  # expensive for n >= 10_000;
                        self.n.reshape(1, -1),  # line_profiler claims 99.7%
                        COS_THETA)  # experimental complexity O(n^2)
        LFUNCPROD = (COEF * LPMV).sum(axis=1)

        NCOEF = self.n * COEF_RAD
        RAD_COEF = np.hstack([np.zeros((COEF_RAD.shape[0], 1)),
                              NCOEF])
        LFACTOR = np.polynomial.legendre.legval(COS_THETA.flatten(),
                                                RAD_COEF.T,
                                                tensor=False)

        sign_rad = np.sign(self.north_projection(self.p_rad))  # .....
        mag_rad = sign_rad * np.linalg.norm(self.p_rad)
        mag_tan = np.linalg.norm(self.p_tan)  # sign_tan * np.linalg.norm(dp_tan)

        tan_potential = -mag_tan * tan_cosinus * LFUNCPROD

        # correction for below dipole position

        rad_potential = mag_rad * LFACTOR
        potentials = tan_potential + rad_potential
        result = potentials / (4 * np.pi * self.model.conductivity.brain * (self.rz ** 2))
        return result.astype(np.float64)

    def cos_theta(self, ele_versors):
        cos_theta = self.north_projection(ele_versors)

        if np.isnan(cos_theta).any():
            warnings.warn("invalid value of cos_theta", RuntimeWarning)
            cos_theta = np.nan_to_num(cos_theta)

        if (cos_theta > 1).any() or (cos_theta < -1).any():
            warnings.warn("cos_theta out of [-1, 1]", RuntimeWarning)
            cos_theta = np.maximum(-1, np.minimum(1, cos_theta))

        return cos_theta

    def tan_versor_cosinus(self, ele_pos):
        ele_north = self.north_vector(ele_pos)
        ele_parallel = ele_pos - ele_north
        ele_parallel_v = ele_parallel / np.sqrt(np.square(ele_parallel).sum(axis=1).reshape(-1, 1))

        tan_parallel = self.p_tan - self.north_vector(self.p_tan)
        tan_r = np.sqrt(np.square(tan_parallel).sum())
        if tan_r == 0:
            warnings.warn("no tangential dipole",
                          RuntimeWarning)
            return np.zeros((ele_pos.shape[0], 1))

        tan_parallel_v = tan_parallel / tan_r
        cos = np.matmul(ele_parallel_v,
                        tan_parallel_v.T)

        if abs(cos).max() - 1 > 1e-10:
            warnings.warn("cos out of [-1 - 1e-10, 1 + 1e-10]",
                          RuntimeWarning)

        if np.isnan(cos).any():
            warnings.warn("invalid value of cos", RuntimeWarning)
            cos = np.nan_to_num(cos)

        return cos

    def H_v(self, r_ele, rad=False):
        if self.model.precision == "float128":
            COEF = np.full((len(r_ele), len(self.n)),
                           np.nan, dtype=np.float128)
        else:
            COEF = np.full((len(r_ele), len(self.n)),
                           np.nan)

        IDX_BELOW = r_ele < self.loc_r

        if IDX_BELOW.any():
            warnings.warn(
                "trying to sample analytically solved potential in undefined areas, expect NaNs in the solution")

        # TODO: fix the below sampling...
        # if IDX_BELOW.any():
        #     _r_ele = r_ele[IDX_BELOW].reshape(-1, 1)
        #     T1 = ((_r_ele / self.radius.brain) ** self.n) * self.A1()
        #
        #     if rad:
        #         T2 = -1 * ((_r_ele / self.rz) ** (
        #                 self.n - 1))
        #     else:
        #         T2 = ((_r_ele / self.rz) ** (
        #                 self.n + 1))
        #     COEF[IDX_BELOW, :] = T1 + T2

        IDX_LOW = r_ele >= self.loc_r
        IDX_HIGH = r_ele < self.radius.brain
        IDX = IDX_LOW & IDX_HIGH
        if IDX.any():
            _r_ele = r_ele[IDX].reshape(-1, 1)
            T1 = ((_r_ele / self.radius.brain) ** self.n) * self.A1()
            T2 = ((self.rz / _r_ele) ** (
                    self.n + 1))
            COEF[IDX, :] = T1 + T2

        IDX_LOW[IDX_HIGH] = False
        IDX_HIGH = r_ele < self.radius.csf
        IDX = IDX_LOW & IDX_HIGH
        if IDX.any():
            _r_ele = r_ele[IDX].reshape(-1, 1)
            T1 = ((_r_ele / self.radius.csf) ** self.n) * self.A2()
            T2 = ((self.radius.csf / _r_ele) ** (self.n + 1)) * self.B2()
            COEF[IDX, :] = T1 + T2

        IDX_LOW[IDX_HIGH] = False
        IDX_HIGH = r_ele < self.radius.skull
        IDX = IDX_LOW & IDX_HIGH
        if IDX.any():
            _r_ele = r_ele[IDX].reshape(-1, 1)
            T1 = ((_r_ele / self.radius.skull) ** self.n) * self.A3()
            T2 = ((self.radius.skull / _r_ele) ** (self.n + 1)) * self.B3()
            COEF[IDX, :] = T1 + T2

        IDX_LOW[IDX_HIGH] = False
        IDX_HIGH = r_ele <= self.radius.scalp
        IDX = IDX_LOW & IDX_HIGH
        if IDX.any():
            _r_ele = r_ele[IDX].reshape(-1, 1)
            T1 = ((_r_ele / self.radius.scalp) ** self.n) * self.A4()
            T2 = ((self.radius.scalp / _r_ele) ** (self.n + 1)) * self.B4()
            COEF[IDX, :] = T1 + T2

        if (~IDX_HIGH).any():
            warnings.warn(
                "trying to sample analytically solved potential in undefined areas, expect NaNs in the solution")
        return COEF

    @property
    def n(self):
        return self.model.n

    def A1(self, n=None):
        try:
            return self._A1
        except AttributeError:
            n = self.n

        Z_n = self.Z(n)
        k = (n + 1.) / n
        self._A1 = self.rz1 ** (n + 1) * (Z_n + self.s12 * k) / (self.s12 - Z_n)
        return self._A1

    def A2(self, n=None):
        try:
            return self._A2
        except AttributeError:
            n = self.n

        self._A2 = ((self.A1(n) + self.rz1 ** (n + 1))
                    / (self.Y(n) * self.r21 ** (n + 1) + self.r12 ** n))

        return self._A2

    def A3(self, n=None):
        try:
            return self._A3
        except AttributeError:
            n = self.n

        self._A3 = ((self.A2(n) + self.B2(n))
                    / (self.r23 ** n + self.V(n) * self.r32 ** (n + 1)))
        return self._A3

    def B2(self, n=None):
        try:
            return self._B2
        except AttributeError:
            n = self.n

        self._B2 = self.A2(n) * self.Y(n)
        return self._B2

    def A4(self, n=None):
        try:
            return self._A4
        except AttributeError:
            n = self.n

        k = (n + 1.) / n
        self._A4 = k * ((self.A3(n) + self.B3(n))
                        / (k * self.r34 ** n + self.r43 ** (n + 1)))
        return self._A4

    def B3(self, n=None):
        try:
            return self._B3
        except AttributeError:
            n = self.n

        self._B3 = self.A3(n) * self.V(n)
        return self._B3

    def B4(self, n=None):
        try:
            return self._B4
        except AttributeError:
            n = self.n

        self._B4 = self.A4(n) * n / (n + 1.)
        return self._B4

    def __getattr__(self, name):
        return getattr(self.model, name)
