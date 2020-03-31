#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2020 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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
    from . import test_MeasurementManagerBase, test_Engine
    from ._common import TestCase
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import test_MeasurementManagerBase, test_Engine
    from _common import TestCase

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
        i : object
            `i.id()` is the index of the probed basis function.

        Returns
        -------
        The measurements of `i.id()`-th basis function.
        """
        return self._measurements_of_basis_functions[i.id()]

    @property
    def number_of_measurements(self):
        return len(self._measurements_of_basis_functions[0])

    class Basis(object):
        def __init__(self, id):
            self._id = id

        def id(self):
            return self._id

    def bases(self):
        return list(map(self.Basis,
                        range(len(self._measurements_of_basis_functions))))


class _TrueBasesMatrixMM(_PlainMatrixMM):
    def probe_at_single_point(self, i, j):
        """
        Parameters
        ----------
        i : object
            `i.id()` is the index of the probed basis function.
        j : int
            Index of the measurement point

        Returns
        -------
        j-th measurement of `i.id()`-th basis functions.
        """
        return self._measurements_of_basis_functions[i.id()][j]

    class Basis(_PlainMatrixMM.Basis):
        def __init__(self, id, values, idvector):
            super(_TrueBasesMatrixMM.Basis, self).__init__(id)
            self._values = values
            self._idvector = idvector

        def f(self, i):
            return self._values[i]

        def idvector(self):
            return self._idvector

    def bases(self):
        return [self.Basis(i, values, idvector)
                for i, (values, idvector)
                in enumerate(zip(self._measurements_of_basis_functions,
                                 np.eye(len(self._measurements_of_basis_functions))))]


class TestKernelMatricesOfVerboseFFR(TestCase):
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
            self.checkArrayAlmostEqual(self.PROBED_POTENTIAL_BASIS[:, i] / self.NUMBER_OF_BASIS,
                                       kernels.idvector())
            for j, k in enumerate(k_row):
                self.assertAlmostEqual(k, kernels.f(j))


class MockTestKernelFunctionsOfVerboseFFR(TestKernelFunctionsOfVerboseFFR):
    class MM(TestKernelFunctionsOfVerboseFFR.MM):
        def __init__(self, *args, **kwargs):
            super(MockTestKernelFunctionsOfVerboseFFR.MM,
                  self).__init__(*args, **kwargs)

            self.recorded_calls = collections.defaultdict(list)

        def __enter__(self):
            self.recorded_calls = collections.defaultdict(list)
            return self.recorded_calls

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def probe_at_single_point(self, *args, **kwargs):
            self.recorded_calls['probe_at_single_point'].append((args,
                                                                 kwargs))
            try:
                return super(MockTestKernelFunctionsOfVerboseFFR.MM,
                             self).probe_at_single_point(*args, **kwargs)
            except:
                return 0

    def testMethod_get_kernel_functions_delegatesArbitraryArgumentsTo_probe_at_single_point(self):
        for args, kwargs in [((), {}),
                             ((1, 2), {}),
                             ((), {'a': 12}),
                             ]:

            with self.measurement_mgr as recorded_calls:
                self.reconstructor.get_kernel_functions(*args, **kwargs)
                calls = recorded_calls['probe_at_single_point']

                self.assertEqual(self.NUMBER_OF_BASIS, len(calls))
                for a, kw in calls:
                    self.assertEqual(args, a[1:])
                    self.assertEqual(kwargs, kw)

                self.assertEqual({b.id() for b in self.measurement_mgr.bases()},
                                 {c[0][0].id() for c in calls})


class TestsOfInitializationErrors(test_Engine.TestsOfInitializationErrors):
    CLASS = VerboseFFR

    MM_MISSING_ATTRIBUTE_ERRORS = (
            test_Engine.TestsOfInitializationErrors.MM_MISSING_ATTRIBUTE_ERRORS
            + [('probe_at_single_point', 'ProbeAtSinglePointMethod')])


class TestVerboseMeasurementManagerBase(test_MeasurementManagerBase.TestMeasurementManagerBaseBase):
    def testMethod_probe_at_single_point_isAbstract(self):
        for args, kwargs in [((), {}),
                             ((0,), {}),
                             ((), {'a': 1}),
                             ]:
            with self.assertRaises(NotImplementedError):
                self.manager.probe_at_single_point(None, *args, **kwargs)

    def testHas_MeasurementManagerHasNoProbeAtSinglePointMethodError_TypeErrorAttribute(self):
        self.checkTypeErrorAttribute('MeasurementManagerHasNoProbeAtSinglePointMethodError')

    @property
    def MM_MISSING_ATTRIBUTE_ERRORS(self):
        for row in super(TestVerboseMeasurementManagerBase,
                         self).MM_MISSING_ATTRIBUTE_ERRORS:
            yield row

        yield 'probe_at_single_point', 'ProbeAtSinglePointMethod'


class TestMeasurementManagerBase(test_MeasurementManagerBase.TestMeasurementManagerBase,
                                 TestVerboseMeasurementManagerBase):
    CLASS = VerboseFFR.MeasurementManagerBase


if __name__ == '__main__':
    unittest.main()
