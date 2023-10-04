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


class Stub(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FunctionFieldComponent(Stub):
    def __init__(self, func, fprime):
        super(FunctionFieldComponent,
              self).__init__(func=self._vectorize(func),
                             fprime=self._vectorize(fprime))

    @staticmethod
    def _vectorize(f):
        def wrapper(arg):
            if isinstance(arg, list):
                return list(map(f, arg))

            return f(arg)

        return wrapper


class TestCase(unittest.TestCase):
    def checkArrayEqual(self, expected, observed,
                        cmp=np.testing.assert_array_equal):
        self.assertIsInstance(observed, np.ndarray)
        self.checkArrayLikeEqual(expected, observed, cmp=cmp)

    def checkArrayAlmostEqual(self, expected, observed):
        self.checkArrayEqual(expected, observed,
                             cmp=np.testing.assert_array_almost_equal)

    def checkArrayLikeEqual(self, expected, observed,
                            cmp=np.testing.assert_array_equal):
        self.assertEqual(np.shape(expected),
                         np.shape(observed))
        cmp(expected, observed)

    def checkArrayLikeAlmostEqual(self, expected, observed):
        self.checkArrayLikeEqual(expected, observed,
                                 cmp=np.testing.assert_array_almost_equal)


class SpyKernelSolverClass(object):
    def __init__(self, solution=None):
        self.kernel = None
        self.rhs = None
        self.regularization_parameter = None
        self.call_counter = collections.Counter()
        self.set_solution(solution)

    def set_solution(self, solution):
        self._solution = solution

    def set_leave_one_out_errors(self, errors):
        self._errors = errors

    def __call__(self, kernel):
        self.call_counter['__init__'] += 1
        self.kernel = kernel
        return self.Spy(self)

    def _call(self, rhs, regularization_parameter=None):
        self.call_counter['__call__'] += 1
        self._function_of_rhs_and_regularization_parameter(regularization_parameter, rhs)
        return self._solution

    def _leave_one_out_errors(self, rhs, regularization_parameter=None):
        self.call_counter['leave_one_out_errors'] += 1
        self._function_of_rhs_and_regularization_parameter(regularization_parameter, rhs)
        return self._errors

    def _function_of_rhs_and_regularization_parameter(self, regularization_parameter, rhs):
        self.rhs = rhs
        self.regularization_parameter = regularization_parameter

    class Spy(object):
        def __init__(self, parent):
            self._parent = parent

        def __call__(self, *args, **kwargs):
            return self._parent._call(*args, **kwargs)

        def leave_one_out_errors(self, *args, **kwargs):
            return self._parent._leave_one_out_errors(*args, **kwargs)