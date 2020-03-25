#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2020 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
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

try:
    from ._common import TestCase
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    from _common import TestCase

from kesi._verbose import _Eigenreconstructor


class _ReconstructorStub(object):
    def __init__(self, parent, eigenvalues, eigenvectors):
        self.parent = parent
        self._EIGENVALUES = np.array(eigenvalues)
        self._EIGENVECTORS = np.array(eigenvectors)

    def __call__(self, measurements, solution):
        self._measurements = measurements
        self._solution = solution
        return self

    def __enter__(self):
        self._measurement_vector_called = 0
        self._wrap_kernel_solution_called = 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent.assertEqual(1, self._measurement_vector_called)
        self.parent.assertEqual(1, self._wrap_kernel_solution_called)

    @property
    def kernel(self):
        return np.matmul(self._EIGENVECTORS,
                         np.matmul(np.diagflat(self._EIGENVALUES),
                                   self._EIGENVECTORS.T))

    def _wrap_kernel_solution(self, solution):
        self._wrap_kernel_solution_called += 1
        self.parent.checkArrayAlmostEqual(self._solution,
                                          solution.reshape(np.shape(self._solution)))
        return self

    def _measurement_vector(self, measurements):
        self._measurement_vector_called += 1
        self.parent.assertIs(measurements, self._measurements)
        return np.reshape(measurements, (-1, 1))


class _TestEigenreconstructorGivenEigenvectorsBase(TestCase):
    @property
    def EIGENVECTORS(self):
        return np.array(self._EIGENVECTORS)

    def setUp(self):
        if not hasattr(self, 'EIGENVALUES'):
            self.skipTest('test in abstract class called')
            return

        self.rec = _ReconstructorStub(self,
                                      self.EIGENVALUES,
                                      self.EIGENVECTORS)
        self.reconstructor = _Eigenreconstructor(self.rec)

    def testHasEigenvalues(self):
        self.checkArrayLikeAlmostEqual(sorted(self.EIGENVALUES),
                                       sorted(self.reconstructor.EIGENVALUES))

    def testCalledWithoutParametersDividesEigenvectorsByEigenvalues(self):
        for eigenvalue, eigenvector in zip(self.EIGENVALUES,
                                           self.EIGENVECTORS.T):
            self.checkSolution(eigenvector / eigenvalue, eigenvector)

    def testCalledWithoutParametersSolvesEigenvectorMixture(self):
        rhs, solution = 0, 0
        for i, (eigenvalue, eigenvector) in enumerate(zip(self.EIGENVALUES,
                                                          self.EIGENVECTORS.T),
                                                      start=1):
            rhs += i * eigenvector
            solution += i * eigenvector / eigenvalue

        self.checkSolution(solution, rhs)

    def testCalledWithMaskZeroesMaskedEigenvalues(self):
        n = len(self.EIGENVALUES)
        for n_components in range(n):
            mask = np.arange(n) <= n_components
            for i, (m, eigenvalue, eigenvector) in enumerate(zip(mask,
                                                                 self.EIGENVALUES,
                                                                 self.EIGENVECTORS.T)):
                self.checkSolution(m * eigenvector / eigenvalue, eigenvector, mask=mask)

    def testCalledWithMaskSolvesEigenvectorMixture(self):
        n = len(self.EIGENVALUES)
        for n_components in range(n):
            mask = np.arange(n) <= n_components
            rhs, solution = 0, 0
            for i, (m, eigenvalue, eigenvector) in enumerate(zip(mask,
                                                                 self.EIGENVALUES,
                                                                 self.EIGENVECTORS.T),
                                                             start=1):
                rhs += i * eigenvector
                solution += m * i * eigenvector / eigenvalue

            self.checkSolution(solution, rhs, mask=mask)

    def checkSolution(self, solution, rhs, **kwargs):
        with self.rec(measurements=rhs,
                      solution=solution):
            self.assertIs(self.rec,
                          self.reconstructor(rhs, **kwargs))


class TestEigenreconstructorGivenIdentityEigenvectors(
          _TestEigenreconstructorGivenEigenvectorsBase):
    EIGENVALUES = [2., 4.]
    _EIGENVECTORS = [[1., 0.],
                     [0., 1.]]


class TestEigenreconstructorGivenNonidentityEigenvectors(
          _TestEigenreconstructorGivenEigenvectorsBase):
    EIGENVALUES = [2., 4.]
    _EIGENVECTORS = [[1., 1.],
                     [-1., 1.]] * np.sqrt([0.5])


if __name__ == '__main__':
    unittest.main()
