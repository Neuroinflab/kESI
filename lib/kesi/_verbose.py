#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################

import numpy as np

from ._engine import FunctionalFieldReconstructor

class VerboseFFR(FunctionalFieldReconstructor):
    """
    Attributes
    ----------
    PHI
    K
    M
    N
    kernel

    Methods
    -------
    PHI_TILDE(measurement_manager)
    K_TILDE(measurement_manager)
    cross_kernel(measurement_manager)

    Notes
    -----
    This class extends its parent, providing access to theoretical concepts
    presented in our publication [1]_, thus some names follow convention used
    therein.

    .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
       reliable current source density estimation" (preprint available at
       `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
       doi: 10.1101/708511
    """

    @property
    def PHI(self):
        r"""
        A matrix of basis functions (rows) probed at measurement points
        (columns).

        The matrix is not normalized.

        Returns
        -------
        numpy.ndarray
        The matrix as M x N numpy array.

        See also
        --------
        M, N

        Notes
        -----
        The measurement manager may affect the returned value.

        The matrix is denormalized as parental class uses normalized one.
        Thus it may be affected by numerical errors.

        `PHI[:, i]` is :math:`\Phi(x_i)` (see eq. 16 and above in [1]_), where
        :math:`x_i` is the i-th measurement point(electrode).

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        return self._pre_kernel * self.M

    @property
    def K(self):
        r"""
        The kernel matrix.

        The matrix is not normalized.

        Returns
        -------
        numpy.ndarray
            The kernel matrix as an N x N numpy array.

        See also
        --------
        N, PHI

        Notes
        -----
        The measurement manager may affect the returned value.

        The kernel matrix is denormalized as parental class uses normalized one.
        Thus it may be affected by numerical errors.

        The kernel matrix (:math:`K`) is defined in [1]_ (see eq. 16 and 25).
        `K == PHI.T @ PHI`

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        return self.kernel * self.M

    @property
    def kernel(self):
        r"""
        The actual kernel matrix.

        The normalized kernel matrix used by the object.

        Returns
        -------
        numpy.ndarray
            The kernel matrix as an N x N numpy array.

        See also
        --------
        K, N, M

        Notes
        -----
        The measurement manager may affect the returned value.

        The returned kernel matrix may change to reflect implementation changes
        in the parental class.

        `kernel == K / M`
        """
        return self._kernel

    @property
    def M(self):
        r"""
        The number of basis functions.

        Returns
        -------
        int
        """
        return self._pre_kernel.shape[0]

    @property
    def N(self):
        r"""
        The number of measurement points (electrodes).

        Returns
        -------
        int
        """
        return self._pre_kernel.shape[1]

    def PHI_TILDE(self, measurement_manager):
        r"""
        A matrix of basis functions (rows) probed at estimation points
        (columns).

        The matrix is not normalized.

        Parameters
        ----------
        measurement_manager : instance of kesi.MeasurementManagerBase subclass
            The measurement manager is an object implementing `.probe(basis)`
            method, which probes appropriate function related to `basis`
            at appropriate estimation points and returns sequence of values.
            The number of the estimation points is given by its
            `.number_of_measurements` attribute.

        Returns
        -------
        PHI_TILDE : numpy.ndarray
            The matrix as a `measurement_manager.number_of_measurements` x `M`
            numpy array.

        See also
        --------
        M

        Notes
        -----
        The measurement manager may affect the returned value.

        `PHI_TILDE[:, i]` is :math:`\tilde{Phi}(x_i)` (see eq. above 16 in [1]_),
        where :math:`x_i` is the i-th estimation point.

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        PHI_TILDE = np.empty((self.M,
                              measurement_manager.number_of_measurements))
        self._fill_probed_components(PHI_TILDE, measurement_manager.probe)
        return PHI_TILDE

    def K_TILDE(self, measurement_manager):
        r"""
        A cross-kernel matrix.

        The matrix is not normalized.

        Parameters
        ----------
        measurement_manager : instance of kesi.MeasurementManagerBase subclass
            The measurement manager is an object implementing `.probe(basis)`
            method, which probes appropriate function related to `basis`
            at appropriate estimation points and returns sequence of values.
            The number of the estimation points is given by its
            `.number_of_measurements` attribute.

        Returns
        -------
        K_TILDE : numpy.ndarray
            The cross-kernel matrix as a
            `measurement_manager.number_of_measurements` x `N` numpy array.

        See also
        --------
        N, PHI, PHI_TILDE(measurement_manager)

        Notes
        -----
        Measurement managers may affect the returned value.

        The cross-kernel matrix (:math:`\tilde{K}`) is defined in [1]_
        (see eq. above 27) analogously to :math:`K` (eq 25).
        `K_TILDE == PHI_TILDE(measurement_manager).T @ PHI`

        The cross-kernel matrix is calculated using denormalized PHI matrix
        as parental class uses normalized PHI one.  Thus it may be affected by
        numerical errors.

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        return np.matmul(self.PHI_TILDE(measurement_manager).T,
                         self.PHI)

    def cross_kernel(self, measurement_manager):
        r"""
        The actual cross-kernel matrix.

        The matrix is normalized and may be used with the actual kernel matrix
        used by the object.

        Parameters
        ----------
        measurement_manager : instance of kesi.MeasurementManagerBase subclass
            The measurement manager is an object implementing `.probe(basis)`
            method, which probes appropriate function related to `basis`
            at appropriate estimation points and returns sequence of values.
            The number of the estimation points is given by its
            `.number_of_measurements` attribute.

        Returns
        -------
        cross_kernel : numpy.ndarray
            The cross-kernel matrix as a
            `measurement_manager.number_of_measurements` x `N` numpy array.

        See also
        --------
        N, PHI, PHI_TILDE(measurement_manager)

        Notes
        -----
        Measurement managers may affect the returned value.

        `cross_kernel == K_TILDE(measurement_manager) / M`
        """
        return np.matmul(self.PHI_TILDE(measurement_manager).T,
                         self._pre_kernel)
