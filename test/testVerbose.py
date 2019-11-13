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

import unittest
import numpy as np

from kesi._engine import FunctionalFieldReconstructor, MeasurementManagerBase
from kesi._verbose import VerboseFFR


class _MatrixMM(MeasurementManagerBase):
    def __init__(self, measurements_of_basis_functions):
        '''
        Parameters
        ----------
        measurements_of_basis_functions : 2D matrix of floats
            A matrix of measurements of basis functions (rows) at measurement
            points (columns).
        '''
        self._set(measurements_of_basis_functions)

    def _set(self, measurements_of_basis_functions):
        self._measurements_of_basis_functions = measurements_of_basis_functions

    def probe(self, i):
        """
        Parameters
        ----------
        i : int
            Index of the probed basis function.

        Returns
        -------
        i-th of the measurements of basis functions.
        """
        return self._measurements_of_basis_functions[i]

    @property
    def number_of_measurements(self):
        return len(self._measurements_of_basis_functions[0])


class TestKernelsOfVerboseFFR(unittest.TestCase):
    _PHI = [[1, 2],
            [3, 4],
            [5, 6]]
    _PHI_TILDE = [[-10, 20],
                  [-30, 40],
                  [-50, 60]]

    @property
    def PHI(self):
        return np.array(self._PHI,
                        dtype=float)

    @property
    def PHI_TILDE(self):
        return np.array(self._PHI_TILDE,
                        dtype=float)

    @property
    def K(self):
        return np.matmul(np.transpose(self.PHI),
                         self.PHI)

    @property
    def K_TILDE(self):
        return np.matmul(np.transpose(self.PHI_TILDE),
                         self.PHI)

    @property
    def M(self):
        return self.PHI.shape[0]

    @property
    def N(self):
        return self.PHI.shape[1]

    def setUp(self):
        self.estimation_mgr = _MatrixMM(self.PHI_TILDE)
        self.reconstructor = self.getReconstructor(self.PHI)

    def getReconstructor(self, measurements_of_basis_functions):
        return VerboseFFR(range(len(measurements_of_basis_functions)),
                          _MatrixMM(measurements_of_basis_functions))

    def checkArrayEqual(self, expected, observed):
        self.assertIsInstance(observed, np.ndarray)
        self.assertEqual(np.shape(expected),
                         observed.shape)
        np.testing.assert_array_equal(expected,
                                      observed)

    def checkArrayAlmostEqual(self, expected, observed):
        self.assertIsInstance(observed, np.ndarray)
        self.assertEqual(np.shape(expected),
                         observed.shape)
        np.testing.assert_array_almost_equal(expected,
                                             observed)

    def testIsSubclassOfFunctionalFieldReconstructor(self):
        self.assertTrue(issubclass(VerboseFFR,
                                   FunctionalFieldReconstructor))

    def testAttribute_PHI(self):
        self.checkArrayAlmostEqual(self.PHI,
                                   self.reconstructor.PHI)

    def testAttribute_K(self):
        self.checkArrayAlmostEqual(self.K,
                                   self.reconstructor.K)

    def testAttribute_kernel(self):
        self.checkArrayAlmostEqual(self.K / self.M,
                                   self.reconstructor.kernel)

    def testAttribute_M(self):
        self.assertIsInstance(self.reconstructor.M, int)
        self.assertEqual(self.M,
                         self.reconstructor.M)

    def testAttribute_N(self):
        self.assertIsInstance(self.reconstructor.N, int)
        self.assertEqual(self.N,
                         self.reconstructor.N)

    def testMethod_PHI_TILDE(self):
        self.checkArrayEqual(self.PHI_TILDE,
                             self.reconstructor.PHI_TILDE(self.estimation_mgr))

    def testMethod_K_TILDE(self):
        self.checkArrayAlmostEqual(self.K_TILDE,
                                   self.reconstructor.K_TILDE(self.estimation_mgr))

    def testMethod_cross_kernel(self):
        self.checkArrayAlmostEqual(self.K_TILDE / self.M,
                                   self.reconstructor.cross_kernel(self.estimation_mgr))


if __name__ == '__main__':
    unittest.main()
