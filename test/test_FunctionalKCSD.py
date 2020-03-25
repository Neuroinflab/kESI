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

try:
    from ._common import FunctionFieldComponent
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    from _common import FunctionFieldComponent

try:
    import pandas as pd
except:
    pd = None

import kesi as kesi

class _GivenComponentsAndNodesBase(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, 'FIELD_COMPONENTS'):
            self.skipTest('test in abstract class called')

    def createField(self, name, points, weights={}):
        return {k: sum(getattr(f, name)([k])[0] * weights.get(c, 1)
                       for c, f in self.FIELD_COMPONENTS.items())
                for k in points
                }

    def createReconstructor(self, name, nodes):
        return kesi.FunctionalKernelFieldReconstructor(
                        self.FIELD_COMPONENTS.values(),
                        name,
                        nodes)

    def _checkApproximation(self, expected, measured, measuredName,
                            regularization_parameter=None):
        reconstructor = self.createReconstructor(measuredName,
                                                 list(measured))
        self.checkResultsAlmostEqual(self._getApproximator(
                                              reconstructor,
                                              measured,
                                              regularization_parameter),
                                     expected)

    def _getApproximator(self, reconstructor, measured,
                         regularization_parameter=None):
        if regularization_parameter is None:
            return reconstructor(measured)

        return reconstructor(measured,
                             regularization_parameter=regularization_parameter)

    def checkWeightedApproximation(self, measuredName, nodes,
                                   names, points, weights={}):
        self._checkApproximation(
            {name: self.createField(name, points, weights=weights)
             for name in names},
            self.createField(measuredName, nodes, weights=weights),
            measuredName)

    def checkReconstructor(self, expected, reconstructor, funcValues,
                           regularization_parameter=None):
        self.checkResultsAlmostEqual(self._getApproximator(
                                              reconstructor,
                                              funcValues,
                                              regularization_parameter),
                                     expected)

    def checkResultsAlmostEqual(self, approximator, expected):
        for name in expected:
            field = getattr(approximator, name)

            for k, v in expected[name].items():
                self.assertAlmostEqual(v, field(k))


class _GivenSingleComponentSingleNodeBase(_GivenComponentsAndNodesBase):
    NODES = ['zero']

    def testProperlyHandlesTheNode(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        self.NODES,
                                        weights={'1': 2})

    def testExtrapolatesOneOtherPoint(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        points = ['one']
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        points)

    def testExtrapolatesManyOtherPoints(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        points = ['one', 'two']
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        points)


class _GivenSingleConstantComponent(object):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        }


class GivenSingleConstantFieldComponentSingleNode(_GivenSingleComponentSingleNodeBase,
                                                  _GivenSingleConstantComponent):
    pass


class _GivenTwoNodesBase(_GivenComponentsAndNodesBase):
    NODES = ['zero', 'two']

    def testProperlyHandlesTheNodes(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        self.NODES,
                                        weights={'1': 2})

    def testInterpolates(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        ['one'],
                                        weights={'1': 2})

    def testExtrapolatesAndInterpolates(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        ['one', 'three'],
                                        weights={'1': 2})


class _GivenTwoNodesAndConstantComponentsTestLeaveOneOutBase(_GivenComponentsAndNodesBase):
    NODES = ['zero', 'one']

    def setUp(self):
        super(_GivenTwoNodesAndConstantComponentsTestLeaveOneOutBase,
              self).setUp()
        self.reconstructor = self.createReconstructor('func', self.NODES)

    def checkLeaveOneOut(self, expected, measurements,
                         regularization_parameter):
        observed = self.reconstructor.leave_one_out_errors(
                                          measurements,
                                          regularization_parameter=regularization_parameter)
        self.assertEqual(len(expected), len(observed))
        for e, o in zip(expected,
                        observed):
            try:
                self.assertAlmostEqual(e, o)
            except TypeError:
                raise AssertionError(repr(o))

    def testLeaveOneOutErrorsGivenConstantInputAndNoRegularisation(self):
        self.checkLeaveOneOut([0, 0],
                              {'zero': 1,
                               'one': 1,
                               },
                              0)

    def testLeaveOneOutErrorsGivenConstantInputAndRegularisation(self):
        expected = [-0.5, -0.5]
        self.checkLeaveOneOut(expected,
                              {'zero': 1,
                               'one': 1,
                              },
                              1)

    def testLeaveOneOutErrorsGivenVariableInputAndNoRegularisation(self):
        self.checkLeaveOneOut([-1, 1],
                              {'zero': 2,
                               'one': 1,
                               },
                              0)


class GivenTwoNodesAndOneConstantComponentTestLeaveOneOut(_GivenTwoNodesAndConstantComponentsTestLeaveOneOutBase,
                                           _GivenSingleConstantComponent):
    pass


class GivenTwoNodesAndTwoSameConstantComponentsTestLeaveOneOut(_GivenTwoNodesAndConstantComponentsTestLeaveOneOutBase):
     FIELD_COMPONENTS = {'a': FunctionFieldComponent(lambda x: 1,
                                                     lambda x: 0),
                         'b': FunctionFieldComponent(lambda x: 1,
                                                     lambda x: 0),
                         }



class GivenTwoNodesAndTwoLinearFieldComponents(_GivenTwoNodesBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        'x': FunctionFieldComponent({'zero': 0,
                                                     'one': 1,
                                                     'two': 2,
                                                     'three': 3}.get,
                                                    {'zero': 1,
                                                     'one': 1,
                                                     'two': 1,
                                                     'three': 1}.get)}

    def testRegularisation(self):
        expected = {'func': {'zero': 0.8,
                             'one': 1.4,
                             },
                    'fprime': {'zero': 0.6,
                               'one': 0.6,
                               },
                    }
        self.checkReconstructor(expected,
                                self.createReconstructor(
                                        'func',
                                        list(expected['func'])),
                                {'zero': 1, 'one': 2},
                                regularization_parameter=0.5)


class GivenTwoNodesAndThreeLinearFieldComponents(_GivenTwoNodesBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        'x': FunctionFieldComponent({'zero': 0,
                                                     'one': 1,
                                                     'two': 2,
                                                     'three': 3}.get,
                                                    {'zero': 1,
                                                     'one': 1,
                                                     'two': 1,
                                                     'three': 1}.get),
                        '1 - x': FunctionFieldComponent({'zero': 1,
                                                         'one': 0,
                                                         'two': -1,
                                                         'three': -2}.get,
                                                        {'zero': -1,
                                                         'one': -1,
                                                         'two': -1,
                                                         'three': -1}.get),
                        }


@unittest.skipIf(pd is None, 'No pandas module')
class WhenCalledWithPandasSeries(GivenTwoNodesAndThreeLinearFieldComponents):
    def createField(self, name, points, weights={}):
        return pd.Series(super(WhenCalledWithPandasSeries,
                               self).createField(name,
                                                 points,
                                                 weights=weights))

    def _checkApproximation(self, expected, measured, measuredName,
                            regularization_parameter=None):
        reconstructor = self.createReconstructor(
                              measuredName,
                              list(measured.index))
        approximator = self._getApproximator(reconstructor,
                                             measured,
                                             regularization_parameter)
        for name in expected:
            field = getattr(approximator, name)
            for k in expected[name].index:
                self.assertAlmostEqual(expected[name][k],
                                       field(k))


if __name__ == '__main__':
    unittest.main()
