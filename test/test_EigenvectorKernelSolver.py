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


from kesi._engine import _EigenvectorKernelSolver


class _TestEigenvectorKernelSolverBase(TestCase):
    @property
    def EIGENVECTORS(self):
        return np.array(self._EIGENVECTORS)

    @property
    def EIGENVALUES(self):
        return np.array(self._EIGENVALUES)

    @property
    def SIGMA(self):
        return np.diag(self.EIGENVALUES)

    @property
    def KERNEL(self):
        return np.matmul(self.EIGENVECTORS,
                         np.matmul(self.SIGMA,
                                   self.EIGENVECTORS.T))

    def setUp(self):
        if not hasattr(self, 'EIGENVALUES'):
            self.skipTest('test in abstract class called')
            return

        self.solver = _EigenvectorKernelSolver(self.KERNEL)

    def testExtractsEigenvalues(self):
        self.checkArrayLikeAlmostEqual(sorted(self.EIGENVALUES),
                                       sorted(self.solver.EIGENVALUES))

    def testExtractsEigenvectors(self):
        # Neither order nor sign of eigenvectors are defined
        E_IDX = np.argsort(self.EIGENVALUES)
        O_IDX = np.argsort(self.solver.EIGENVALUES)
        EXPECTED = self.EIGENVECTORS[:, E_IDX]
        OBSERVED = self.solver.EIGENVECTORS[:, O_IDX]
        self.checkArrayAlmostEqual(np.identity(len(E_IDX)),
                                   abs(np.matmul(EXPECTED.T, OBSERVED)))

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
                                                                 self.solver.EIGENVALUES,
                                                                 self.solver.EIGENVECTORS.T)):
                self.checkSolution(m * eigenvector / eigenvalue, eigenvector, mask=mask)

    def testCalledWithMaskSolvesEigenvectorMixture(self):
        n = len(self.EIGENVALUES)
        for n_components in range(n):
            mask = np.arange(n) <= n_components
            rhs, solution = 0, 0
            for i, (m, eigenvalue, eigenvector) in enumerate(zip(mask,
                                                                 self.solver.EIGENVALUES,
                                                                 self.solver.EIGENVECTORS.T),
                                                             start=1):
                rhs += i * eigenvector
                solution += m * i * eigenvector / eigenvalue

            self.checkSolution(solution, rhs, mask=mask)

    def testSolvesManyRightHandSidesAtOnce(self):
        SCALED = np.matmul(self.EIGENVECTORS, np.diag(1. / self.EIGENVALUES))
        self.checkSolution(np.hstack((SCALED, np.cumsum(SCALED, axis=1))),
                           np.hstack((self.EIGENVECTORS,
                                      np.cumsum(self.EIGENVECTORS, axis=1))))

    def checkSolution(self, expected, rhs, **kwargs):
        self.checkArrayAlmostEqual(expected,
                                   self.solver(rhs, **kwargs))


class TestGivenIdentityKernel(_TestEigenvectorKernelSolverBase):
    _EIGENVECTORS = [[1, 0],
                     [0, 1]]
    _EIGENVALUES = [1, 1]


class TestGivenDiagonalKernel(_TestEigenvectorKernelSolverBase):
    _EIGENVECTORS = [[0, 1],
                     [1, 0]]
    _EIGENVALUES = [4, 0.25]


class TestGivenNontrivialKernel(_TestEigenvectorKernelSolverBase):
    _EIGENVECTORS = [[2**-0.5, 2**-0.5],
                     [2**-0.5, -2**-0.5]]
    _EIGENVALUES = [4, 0.25]


if __name__ == '__main__':
    unittest.main()
