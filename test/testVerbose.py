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
import collections

import numpy as np

try:
    from . import testMeasurementManagerBase, testEngine
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import testMeasurementManagerBase, testEngine

from kesi._engine import FunctionalFieldReconstructor
from kesi._verbose import VerboseFFR


class _PlainMatrixMM(VerboseFFR.MeasurementManagerBase):
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

    def bases(self):
        return range(len(self._measurements_of_basis_functions))


class _TrueBasesMatrixMM(_PlainMatrixMM):
    def probe_at_single_point(self, i, j):
        """
        Parameters
        ----------
        i : int
            Index of the probed basis function.
        j : int
            Index of the measurement point

        Returns
        -------
        j-th measurement of i-th basis functions.
        """
        return self._measurements_of_basis_functions[i][j]

    class Basis(int):
        def __new__(cls, values, *args, **kwargs):
            return super(_TrueBasesMatrixMM.Basis, cls).__new__(cls, *args, **kwargs)

        def __init__(self, values, *args, **kwargs):
            # super(_TrueBasesMatrixMM.Basis, self).__init__()
            self._values = values

        def f(self, i):
            return self._values[i]

    def bases(self):
        return [self.Basis(values, i)
                for i, values
                in enumerate(self._measurements_of_basis_functions)]


class TestKernelMatricesOfVerboseFFR(unittest.TestCase):
    MM = _PlainMatrixMM

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
        self.estimation_mgr = self.MM(self.PROBED_CSD_BASIS)
        self.reconstructor = self.getReconstructor(self.PROBED_POTENTIAL_BASIS)

    def getReconstructor(self, measurements_of_basis_functions):
        self.measurement_mgr = self.MM(measurements_of_basis_functions)
        return VerboseFFR(self.measurement_mgr.bases(),
                          self.measurement_mgr)


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


class TestKernelFunctionsOfVerboseFFR(TestKernelMatricesOfVerboseFFR):
    MM = _TrueBasesMatrixMM

    def testMethod_get_kernel_functions(self):
        for i, k_row in enumerate(self.KERNEL):
            kernels = self.reconstructor.get_kernel_functions(i)
            for j, k in enumerate(k_row):
                self.assertAlmostEqual(k, kernels.f(j))


class MockTestKernelFunctionsOfVerboseFFR(TestKernelMatricesOfVerboseFFR):
    class MM(TestKernelMatricesOfVerboseFFR.MM):
        def __init__(self, *args, **kwargs):
            super(MockTestKernelFunctionsOfVerboseFFR.MM,
                  self).__init__(*args, **kwargs)

            self.recorded_calls = collections.defaultdict(list)

        def probe_at_single_point(self, field, *args, **kwargs):
            self.recorded_calls['probe_at_single_point'].append((field,
                                                                 args,
                                                                 kwargs))
            return 0

    def testMethod_get_kernel_functions_delegatesArbitraryArgumentsTo_probe_at_single_point(self):
        for args, kwargs in [((), {}),
                             ((1, 2), {}),
                             ((), {'a': 12}),
                             ]:

            calls = self.measurement_mgr.recorded_calls['probe_at_single_point']
            del calls[:] # Python 2.7 does not support list.clear()
            self.reconstructor.get_kernel_functions(*args, **kwargs)

            self.assertEqual(self.NUMBER_OF_BASIS, len(calls))
            for _, a, kw in calls:
                self.assertEqual(args, a)
                self.assertEqual(kwargs, kw)
            basis_probed = [c[0] for c in calls]
            for basis in self.reconstructor._field_components:
                self.assertIn(basis, basis_probed)
            for basis in basis_probed:
                self.assertIn(basis, self.reconstructor._field_components)


class TestsOfInitializationErrors(testEngine.TestsOfInitializationErrors):
    CLASS = VerboseFFR

    MM_MISSING_ATTRIBUTE_ERRORS = (
        testEngine.TestsOfInitializationErrors.MM_MISSING_ATTRIBUTE_ERRORS
        + [('probe_at_single_point', 'ProbeAtSinglePointMethod')])



class TestMeasurementManagerBase(testMeasurementManagerBase.TestMeasurementManagerBase):
    CLASS = VerboseFFR.MeasurementManagerBase

    def testMethod_probe_at_single_point_isAbstract(self):
        for args, kwargs in [((), {}),
                             ((0,), {}),
                             ((), {'a': 1}),
                             ]:
            with self.assertRaises(NotImplementedError):
                self.manager.probe_at_single_point(None, *args, **kwargs)


if __name__ == '__main__':
    unittest.main()
