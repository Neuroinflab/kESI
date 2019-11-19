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
    _PROBED_POTENTIAL_BASIS = [[1, 2],
                               [3, 4],
                               [5, 6]]
    _PROBED_CSD_BASIS = [[-10, 20],
                         [-30, 40],
                         [-50, 60]]

    @property
    def PROBED_POTENTIAL_BASIS(self):
        return np.array(self._PROBED_POTENTIAL_BASIS,
                        dtype=float)

    @property
    def PROBED_CSD_BASIS(self):
        return np.array(self._PROBED_CSD_BASIS,
                        dtype=float)

    @property
    def KERNEL(self):
        return np.matmul(np.transpose(self.PROBED_POTENTIAL_BASIS),
                         self.PROBED_POTENTIAL_BASIS) / self.NUMBER_OF_BASIS

    @property
    def CROSS_KERNEL(self):
        return np.matmul(np.transpose(self.PROBED_CSD_BASIS),
                         self.PROBED_POTENTIAL_BASIS) / self.NUMBER_OF_BASIS

    @property
    def NUMBER_OF_BASIS(self):
        return self.PROBED_POTENTIAL_BASIS.shape[0]

    @property
    def NUMBER_OF_ELECTRODES(self):
        return self.PROBED_POTENTIAL_BASIS.shape[1]

    def setUp(self):
        self.estimation_mgr = _MatrixMM(self.PROBED_CSD_BASIS)
        self.reconstructor = self.getReconstructor(self.PROBED_POTENTIAL_BASIS)

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

    def testAttribute_probed_basis(self):
        self.checkArrayAlmostEqual(self.PROBED_POTENTIAL_BASIS,
                                   self.reconstructor.probed_basis)

    def testAttribute_kernel(self):
        self.checkArrayAlmostEqual(self.KERNEL,
                                   self.reconstructor.kernel)

    def testAttribute_number_of_basis(self):
        self.assertIsInstance(self.reconstructor.number_of_basis,
                              int)
        self.assertEqual(self.NUMBER_OF_BASIS,
                         self.reconstructor.number_of_basis)

    def testAttribute_number_of_electrodes(self):
        self.assertIsInstance(self.reconstructor.number_of_electrodes,
                              int)
        self.assertEqual(self.NUMBER_OF_ELECTRODES,
                         self.reconstructor.number_of_electrodes)

    def testMethod_get_probed_basis(self):
        self.checkArrayEqual(self.PROBED_CSD_BASIS,
                             self.reconstructor.get_probed_basis(self.estimation_mgr))

    def testMethod_get_kernel_matrix(self):
        self.checkArrayAlmostEqual(self.CROSS_KERNEL,
                                   self.reconstructor.get_kernel_matrix(self.estimation_mgr))


if __name__ == '__main__':
    unittest.main()
