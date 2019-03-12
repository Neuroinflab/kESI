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
from unittest import TestCase

import kesi as kesi

FunctionFieldComponent = collections.namedtuple('FunctionFieldComponent',
                                                ['func', 'fprime'])


class _GivenComponentsAndNodesBase(TestCase):
    def setUp(self):
        if not hasattr(self, 'FIELD_COMPONENTS'):
            self.skipTest('test in abstract class called')

    def createField(self, name, points, weights={}):
        return {k: sum(getattr(f, name)(k) * weights.get(c, 1)
                       for c, f in self.FIELD_COMPONENTS.items())
                for k in points
                }

    def createInterpolator(self, nodes, points):
        return kesi.KernelFieldInterpolator(self.FIELD_COMPONENTS,
                                            nodes=nodes,
                                            points=points)

    def _checkInterpolation(self, expected, interpolatedName, measured,
                           measuredName):
        interpolator = self.createInterpolator({measuredName: list(measured)},
                                               {interpolatedName: list(expected)})
        self.assertEqual(expected,
                         interpolator(interpolatedName, measuredName, measured))

    def checkWeightedInterpolation(self, measuredName, nodes, interpolatedName,
                                   points, weights={}):
        self._checkInterpolation(
            self.createField(interpolatedName, points, weights=weights),
            interpolatedName,
            self.createField(measuredName, nodes, weights=weights),
            measuredName)


class _GivenSingleComponentSingleNodeBase(_GivenComponentsAndNodesBase):
    NODES = ['zero']

    def testProperlyHandlesTheNode(self):
        interpolatedName = 'func'
        measuredName = 'func'
        self.checkWeightedInterpolation(measuredName,
                                        self.NODES,
                                        interpolatedName,
                                        self.NODES,
                                        weights={'1': 2})

    def testExtrapolatesOneOtherPoint(self):
        interpolatedName = 'func'
        measuredName = 'func'
        points = ['one']
        self.checkWeightedInterpolation(measuredName,
                                        self.NODES,
                                        interpolatedName,
                                        points)

    def testExtrapolatesManyOtherPoints(self):
        interpolatedName = 'func'
        measuredName = 'func'
        points = ['one', 'two']
        self.checkWeightedInterpolation(measuredName,
                                        self.NODES,
                                        interpolatedName,
                                        points)


class GivenSingleConstantFieldComponentSingleNode(_GivenSingleComponentSingleNodeBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0)}


class _GivenTwoNodesBase(_GivenComponentsAndNodesBase):
    NODES = ['zero', 'two']

    def testProperlyHandlesTheNodes(self):
        interpolatedName = 'func'
        measuredName = 'func'
        self.checkWeightedInterpolation(measuredName,
                                        self.NODES,
                                        interpolatedName,
                                        self.NODES,
                                        weights={'1': 2})

    def testInterpolates(self):
        interpolatedName = 'func'
        measuredName = 'func'
        self.checkWeightedInterpolation(measuredName,
                                        self.NODES,
                                        interpolatedName,
                                        ['one'],
                                        weights={'1': 2})

    def testExtrapolatesAndInterpolates(self):
        interpolatedName = 'func'
        measuredName = 'func'
        self.checkWeightedInterpolation(measuredName,
                                        self.NODES,
                                        interpolatedName,
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
                                                     'three': 1}.get),
                        }


if __name__ == '__main__':
    unittest.main()