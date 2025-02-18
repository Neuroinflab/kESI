import warnings

import numpy as np
from memoization import cached


class PointMonopole(object):
    def __init__(self, model, monopole_loc, amplitude):
        """Amplitude - float in ampers"""
        self.model = model
        self.amplitude = amplitude
        self.set_monopole_loc(monopole_loc)
        self._set_monopole_r()
        self.n = np.arange(0, self.model.n[-1] + 1)[:, None]

        # we only support placing sources in the first shell
        assert self.loc_r < self.model.radius[0]

        # the first shell is then split to shell below and above the point source
        self.radius = [self.loc_r] + list(self.model.radius)
        self.conductivity = [self.model.conductivity[0], ] + list(self.model.conductivity)
        ### what we need from the model
        # self.model.conductivity
        # self.model.radius
        # self.model.n
        # self.model.precision

    def set_monopole_loc(self, loc):
        self.loc_r = np.sqrt(np.square(loc).sum())

        if self.model.precision == 'float128':
            default_vector = np.array([[0, 0, 1]], dtype=np.float128)
        else:
            default_vector = np.array([[0, 0, 1]])

        self.loc_v = (loc / self.loc_r
                      if self.loc_r != 0
                      else default_vector)

    def _set_monopole_r(self):
        self.rz1 = self.loc_r / self.model.radius.brain

    def __call__(self, X, Y, Z):
        if self.model.precision == 'float128':
            ELECTRODES = np.vstack([X, Y, Z], dtype=np.float128).T
        else:
            ELECTRODES = np.vstack([X, Y, Z]).T

        ele_dist = np.linalg.norm(ELECTRODES, axis=1)
        COS_THETA = self.cos_theta(ELECTRODES / ele_dist.reshape(-1, 1))

        COEFFA = self.COEFFA(ele_dist)
        COEFFB = self.COEFFB(ele_dist)

        COEFFS = COEFFA + COEFFB
        LFACTOR = np.polynomial.legendre.legval(COS_THETA.flatten(),
                                                COEFFS.T,
                                                tensor=False)

        result = self.amplitude / (4 * np.pi * self.model.conductivity.brain) * LFACTOR
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

    def north_projection(self, V):
        return np.matmul(V,
                         self.loc_v.T)

    def COEFFA(self, r_ele):
        """returns an arrauy of coeeficients with size of [n_measurement_points, n], where n is the model order

        Params:
        r_ele - list of electrode distances from center of the shells (0, 0, 0) point)
        """

        if self.model.precision == "float128":
            COEF = np.full((len(r_ele), self.n.shape[0]),
                           np.nan, dtype=np.float128)
        else:
            COEF = np.full((len(r_ele), self.n.shape[0]),
                           np.nan)

        for shell_id, shell_radius in enumerate(self.radius):
            if shell_id == 0:
                in_shell = r_ele < self.loc_r
            else:
                in_shell = np.logical_and((self.radius[shell_id - 1] <= r_ele), (r_ele < shell_radius))
            if in_shell.any():
                if shell_id == 0:
                    COEF[in_shell] = 1
                    COEF[in_shell] = COEF[in_shell] * self.A(1) * ((r_ele[in_shell] / self.radius[1]) ** self.n).T
                elif shell_id == 1:
                    COEF[in_shell] = 1
                    COEF[in_shell] = COEF[in_shell] * self.A(1) * ((r_ele[in_shell] / self.radius[1]) ** self.n).T
                elif shell_id == len(self.radius) - 1:
                    COEF[in_shell] = 0
                else:
                    COEF[in_shell] = 1
                    COEF[in_shell] = COEF[in_shell] * self.A(shell_id) * (
                                (r_ele[in_shell] / self.radius[shell_id]) ** self.n).T
        return COEF

    def COEFFB(self, r_ele):
        """returns an arrauy of coeeficients with size of [n_measurement_points, n], where n is the model order

        Params:
        r_ele - list of electrode distances from center of the shells (0, 0, 0) point)
        """
        if self.model.precision == "float128":
            COEF = np.full((len(r_ele), self.n.shape[0]),
                           np.nan, dtype=np.float128)
        else:
            COEF = np.full((len(r_ele), self.n.shape[0]),
                           np.nan)

        for shell_id, shell_radius in enumerate(self.radius):
            if shell_id == 0:
                in_shell = r_ele < self.loc_r
            else:
                in_shell = np.logical_and((self.radius[shell_id - 1] <= r_ele), (r_ele < shell_radius))
            if in_shell.any():
                if shell_id == 0:
                    COEF[in_shell] = 1 / self.loc_r
                    COEF[in_shell] = COEF[in_shell] * ((r_ele[in_shell] / self.loc_r) ** self.n).T
                elif shell_id == 1:
                    COEF[in_shell] = 1 / self.loc_r
                    COEF[in_shell] = COEF[in_shell] * ((self.loc_r / r_ele[in_shell]) ** (self.n + 1)).T
                elif shell_id == len(self.radius) - 1:
                    COEF[in_shell] = 1
                    COEF[in_shell] = self.B(shell_id) * (
                                (self.radius[shell_id - 1] / r_ele[in_shell]) ** (self.n + 1)).T
                else:
                    COEF[in_shell] = 1
                    COEF[in_shell] = self.B(shell_id) * (
                                (self.radius[shell_id - 1] / r_ele[in_shell]) ** (self.n + 1)).T
        return COEF

    @cached
    def transition_submatric(self, shell_id):
        """Returns n transition matrices, to transition from shell shell_id-1 to shell_id"""
        matrices = []

        for n in self.n[:, 0]:
            A = np.array([[1, 0],
                          [0, self.r(shell_id - 2, shell_id - 1) ** (n + 1)],
                          ]
                         )
            B = np.array([[1, 1],
                          [n, -(n + 1)]
                          ]
                         )
            C = np.array([[1, 0],
                          [0, self.sig(shell_id - 1, shell_id)]
                          ])

            D = np.array([[n + 1, 1],
                          [n, -1]])
            E = np.array([[1 / (self.r(shell_id - 1, shell_id) ** n), 0],
                          [0, 1]
                          ]
                         )
            result = (1 / (2 * n + 1)) * E @ D @ C @ B @ A
            matrices.append(result)
        return matrices

    def r(self, s1, s2):
        """Ratio of shell s1 to shell s2 radii"""
        return self.radius[s1] / self.radius[s2]

    def sig(self, s1, s2):
        """Ratio of shell s1 to shell s2 conductivities"""
        return self.conductivity[s1] / self.conductivity[s2]

    @cached
    def transition_matrix(self, shell_id_from, shell_id_to):
        """Returns n transition matrices in a list, where n is order of the model

        Transition matrix transforms A, B coeff for shell N to A, B coeffs for shell N+1, this function returns combined
        matrices, for multi-steps.
        """

        matrices_to_combine = []
        for shell_id in range(shell_id_from + 1, shell_id_to + 1):
            matrices_to_combine.append(self.transition_submatric(shell_id))

        matrixes = []
        for n in range(self.n.shape[0]):
            if len(matrices_to_combine) == 1:
                matrixes.append(matrices_to_combine[0][n])
            else:
                temporary_matrix = matrices_to_combine[0][n]
                for m in range(1, len(matrices_to_combine)):
                    temporary_matrix = temporary_matrix @ matrices_to_combine[m][n]
                matrixes.append(temporary_matrix)

        return matrixes

    @cached
    def A(self, shell_id):
        """shape [1, n], where n is the order of the model"""
        if shell_id == len(self.radius) - 1:
            A = np.array([0, ] * self.n.shape[0])[None, :]
        elif shell_id == 1:
            M = np.array(self.transition_matrix(1, len(self.radius) - 1))
            A = -M[:, 0, 1] / (M[:, 0, 0] * self.loc_r)[None, :]
        else:
            AB = []
            init = np.array([self.A(1),
                             self.B(1),
                             ]

                            ).T
            for i in range(self.n.shape[0]):
                AB.append(self.transition_matrix(1, shell_id)[i] @ init[i].T)
            A = np.array([i[0, 0] for i in AB])[None, :]
        return A

    @cached
    def B(self, shell_id):
        """shape [1, n], where n is the order of the model"""
        if shell_id == 1:
            B = np.array([1 / self.loc_r, ] * self.n.shape[0])[None, :]
        else:
            AB = []
            init = np.array([self.A(1),
                             self.B(1),
                             ]

                            ).T
            for i in range(self.n.shape[0]):
                AB.append(self.transition_matrix(1, shell_id)[i] @ init[i].T)
            B = np.array([i[1, 0] for i in AB])[None, :]
        return B
