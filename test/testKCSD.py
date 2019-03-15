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

try:
    import pandas as pd
except:
    pd = None

import kesi as kesi

class FunctionFieldComponent(object):
    def __init__(self, func, fprime):
        self._func = func
        self._fprime = fprime

    def func(self, args):
        return [self._func(a) for a in args]

    def fprime(self, args):
        return [self._fprime(a) for a in args]

class _GivenComponentsAndNodesBase(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, 'FIELD_COMPONENTS'):
            self.skipTest('test in abstract class called')

    def createField(self, name, points, weights={}):
        return {k: sum(getattr(f, name)([k])[0] * weights.get(c, 1)
                       for c, f in self.FIELD_COMPONENTS.items())
                for k in points
                }

    def createApproximator(self, nodes, points, lambda_=None):
        if lambda_ is None:
            return kesi.KernelFieldApproximator(self.FIELD_COMPONENTS.values(),
                                                nodes=nodes,
                                                points=points)
        return kesi.KernelFieldApproximator(self.FIELD_COMPONENTS.values(),
                                            nodes=nodes,
                                            points=points,
                                            lambda_=lambda_)

    def _checkApproximation(self, expected, measured,
                            measuredName, lambda_=None):
        approximator = self.createApproximator({measuredName: list(measured)},
                                               {k: list(v)
                                                for k, v in expected.items()},
                                               lambda_=lambda_)
        for name in expected:
            self.assertEqual(expected[name],
                             approximator(name, measuredName, measured))

    def checkWeightedApproximation(self, measuredName, nodes,
                                   names, points, weights={}):
        self._checkApproximation(
            {name: self.createField(name, points, weights=weights)
             for name in names},
            self.createField(measuredName, nodes, weights=weights),
            measuredName)

    def checkApproximator(self, expected, approximator, funcValues):
        for name in expected:
            approximated = approximator(name, 'func', funcValues)
            self.assertEqual(sorted(expected[name]),
                             sorted(approximated))
            for k, v in expected[name].items():
                self.assertAlmostEqual(v, approximated[k])


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


class GivenSingleConstantFieldComponentSingleNode(_GivenSingleComponentSingleNodeBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0)}


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
        self.checkApproximator(expected, self.createApproximator(
            {'func': list(expected['func'])},
            {k: list(v)
             for k, v in expected.items()},
            lambda_=1.0), {'zero': 1, 'one': 2})

    def testCopy(self):
        expected = {'func': {'zero': 0.8,
                             'one': 1.4,
                             },
                    'fprime': {'zero': 0.6,
                               'one': 0.6,
                               },
                    }
        original = self.createApproximator({'func': list(expected['func'])},
                                           {k: list(v)
                                            for k, v in expected.items()},
                                           lambda_=1.0)
        self.checkApproximator(expected, original.copy(),
                               funcValues={'zero': 1, 'one': 2})

    def testCopyRegularisationChange(self):
        expected = {'func': {'zero': 0.8,
                             'one': 1.4,
                             },
                    'fprime': {'zero': 0.6,
                               'one': 0.6,
                               },
                    }
        original = self.createApproximator({'func': list(expected['func'])},
                                           {k: list(v)
                                            for k, v in expected.items()})
        self.checkApproximator(expected, original.copy(lambda_=1.0),
                               funcValues={'zero': 1, 'one': 2})


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


class GivenNumericFunctionsAsFieldComponents(_GivenComponentsAndNodesBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        'x': FunctionFieldComponent(lambda x: x[0],
                                                    lambda x: 1),
                        }

    def testNumpy2DArraysAreValidCollectionsOfTuples(self):
        expected = {'func': {(0,): 1,
                             (1,): 3,
                             },
                    'fprime': {(0,): 2,
                               (1,): 2,
                               },
                    }
        self.checkApproximator(expected,
                               self.createApproximator(
                                    {'func': np.array(list(expected['func']))},
                                    {k: np.array(list(v))
                                     for k, v in expected.items()}),
                                    {(0,): 1,
                                     (1,): 3})

    def testNumpy3DArraysAreValidCollectionsOfNestedTuples(self):
        expected = {'func': {((0,),): 1,
                             ((1,),): 3,
                             },
                    'fprime': {((0,),): 2,
                               ((1,),): 2,
                               },
                    }
        self.checkApproximator(expected,
                               self.createApproximator(
                                    {'func': np.array(list(expected['func']))},
                                    {k: np.array(list(v))
                                     for k, v in expected.items()}),
                                    {((0,),): 1,
                                     ((1,),): 3})



@unittest.skipIf(pd is None, 'No pandas module')
class WhenCalledWithPandasSeries(GivenTwoNodesAndThreeLinearFieldComponents):
    def createField(self, name, points, weights={}):
        return pd.Series(super(WhenCalledWithPandasSeries,
                               self).createField(name,
                                                 points,
                                                 weights=weights))

    def _checkApproximation(self, expected, measured,
                            measuredName, lambda_=None):
        approximator = self.createApproximator({measuredName: list(measured.index)},
                                               {k: list(v.index)
                                                for k, v in expected.items()},
                                               lambda_=lambda_)
        for name in expected:
            approximated = approximator(name, measuredName, measured)
            self.assertIsInstance(approximated, pd.Series)
            self.assertTrue((expected[name] == approximated).all())


if __name__ == '__main__':
    unittest.main()
