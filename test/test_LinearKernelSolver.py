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

from kesi._engine import _LinearKernelSolver

class TestLinearKernelSolver(TestCase):
    def testGivenIdentityKernelIsIdentityFunction(self):
        rhs = np.array([1, 2])
        self.checkSolution(rhs, np.identity(2), rhs)

    def testGivenDiagonalKernelDividesVectorElementsByDiagonal(self):
        diagonal = np.array([0.5, 1, 2])
        rhs = np.array([1, 2, 3])
        self.checkSolution(rhs / diagonal, np.diag(diagonal), rhs)

    def testGivenDiagonalKernelDividesVectorElementsByDiagonal(self):
        diagonal = np.array([0.5, 1, 2])
        rhs = np.array([1, 2, 3])
        self.checkSolution(rhs / diagonal, np.diag(diagonal), rhs)

    def testGivenNontrivialKernelSolvesIt(self):
        self.checkSolution([1, -1],
                           [[1, 0],
                            [1, 1]],
                           [1, 0])

    def testSolvesManyRightHandSidesAtOnce(self):
        self.checkSolution([[1, 1, 0.5],
                            [1, -1, 5]],
                           [[1, 0],
                            [1, 1]],
                           [[1, 1, 0.5],
                            [2, 0, 5.5]])

    def testZeroRegularizationParameterDoesNotAffectTheResult(self):
        self.checkSolution([[1, 1, 0.5],
                            [1, -1, 5]],
                           [[1, 0],
                            [1, 1]],
                           [[1, 1, 0.5],
                            [2, 0, 5.5]],
                           regularization_parameter=0)

    def testRegularizationParameterAffectsTheResult(self):
        self.checkSolution([[1, 1, 0.5],
                            [1, -1, 5]],
                           [[1, 0],
                            [1, 1]],
                           [[2, 2, 1],
                            [3, -1, 10.5]],
                           regularization_parameter=1)

    def checkSolution(self, expected, kernel, rhs, regularization_parameter=None):
        solver = _LinearKernelSolver(np.array(kernel))
        self.checkArrayLikeAlmostEqual(expected,
                                       (solver(rhs)
                                        if regularization_parameter is None
                                        else solver(rhs, regularization_parameter=regularization_parameter)))


if __name__ == '__main__':
    unittest.main()
