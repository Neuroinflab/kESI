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
    from ._common import Stub
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    from _common import Stub

from kesi._engine import FunctionalFieldReconstructor, LinearMixture


class FunctionFieldComponent(Stub):
    def __init__(self, func, fprime):
        super(FunctionFieldComponent,
              self).__init__(func=func,
                             fprime=fprime)


class TestsOfInitializationErrors(unittest.TestCase):
    def testWhenMeasurementManagerLacksAttributesThenRaisesException(self):
        for missing, exception in [('probe',
                                    'ProbeMethod'),
                                   ('load',
                                    'LoadMethod'),
                                   ('number_of_measurements',
                                    'NumberOfMeasurementsAttribute'),
                                   ]:
            measurement_manager = self.getIncompleteMeasurementManager(missing)
            exception_name = 'MeasurementManagerHasNo{}Error'.format(exception)
            for ExceptionClass in [getattr(FunctionalFieldReconstructor,
                                           exception_name),
                                   TypeError]:
                with self.assertRaises(ExceptionClass):
                    FunctionalFieldReconstructor([],
                                                 measurement_manager)

    def getIncompleteMeasurementManager(self, missing):
        return Stub(**{attr: None
                       for attr in ['probe',
                                    'load',
                                    'number_of_measurements']
                       if attr != missing})


class _GivenComponentsAndNodesBase(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, 'FIELD_COMPONENTS'):
            self.skipTest('test in abstract class called')

    def testReturnsLinearMixture(self):
        reconstructor = self.createReconstructor(self.NODES)
        approximator = self._getApproximator(reconstructor,
                                             self.createField(self.NODES),
                                             regularization_parameter=0.1)
        self.assertIsInstance(approximator,
                              LinearMixture)

    def createField(self, points, name='func', weights={}):
        return {k: sum(getattr(f, name)(k) * weights.get(c, 1)
                       for c, f in self.FIELD_COMPONENTS.items())
                for k in points
                }

    def createReconstructor(self, nodes):
        return FunctionalFieldReconstructor(
                        self.FIELD_COMPONENTS.values(),
                        self.MeasurementManager(nodes))

    def _checkApproximation(self, expected, measured,
                            regularization_parameter=None):
        reconstructor = self.createReconstructor(list(measured))
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

    def checkWeightedApproximation(self, nodes, names, points, weights={}):
        self._checkApproximation({name: self.createField(points,
                                                         name,
                                                         weights=weights)
                                  for name in names
                                  },
                                 self.probe(self.createField(nodes,
                                                             weights=weights)))

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


class _GivenMappingAsMeasurements(_GivenComponentsAndNodesBase):
    class MeasurementManager(list):
        def probe(self, field):
            return list(map(field.func, self))

        def load(self, measured):
            return [measured[k] for k in self]

        @property
        def number_of_measurements(self):
            return len(self)

    def probe(self, field):
        """
        Measurements of field.

        The measurements should be returned in a format appropriate for the
        `self.MeasurementManager` class.

        Note
        ----

            This is a stub method to enable use of other MM.
        """
        return field


class _GivenSingleComponentSingleNodeBase(_GivenMappingAsMeasurements):
    NODES = ['zero']

    def testProperlyHandlesTheNode(self):
        approximated = ['func', 'fprime']
        self.checkWeightedApproximation(self.NODES, approximated, self.NODES,
                                        weights={'1': 2})

    def testExtrapolatesOneOtherPoint(self):
        approximated = ['func', 'fprime']
        points = ['one']
        self.checkWeightedApproximation(self.NODES, approximated, points)

    def testExtrapolatesManyOtherPoints(self):
        approximated = ['func', 'fprime']
        points = ['one', 'two']
        self.checkWeightedApproximation(self.NODES, approximated, points)


class _GivenSingleConstantComponent(object):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        }


class GivenSingleConstantFieldComponentSingleNode(_GivenSingleComponentSingleNodeBase,
                                                  _GivenSingleConstantComponent):
    pass


class _GivenTwoNodesBase(object):
    NODES = ['zero', 'two']

    def testProperlyHandlesTheNodes(self):
        approximated = ['func', 'fprime']
        self.checkWeightedApproximation(self.NODES, approximated, self.NODES,
                                        weights={'1': 2})

    def testInterpolates(self):
        approximated = ['func', 'fprime']
        self.checkWeightedApproximation(self.NODES, approximated, ['one'],
                                        weights={'1': 2})

    def testExtrapolatesAndInterpolates(self):
        approximated = ['func', 'fprime']
        self.checkWeightedApproximation(self.NODES, approximated,
                                        ['one', 'three'], weights={'1': 2})


class _GivenTwoNodesAndConstantComponentsTestLeaveOneOutBase(_GivenMappingAsMeasurements):
    NODES = ['zero', 'one']

    def setUp(self):
        super(_GivenTwoNodesAndConstantComponentsTestLeaveOneOutBase,
              self).setUp()
        self.reconstructor = self.createReconstructor(self.NODES)

    def checkLeaveOneOut(self, expected, measurements,
                         regularization_parameter):
        observed = self.reconstructor.leave_one_out_errors(
                                          measurements,
                                          regularization_parameter=regularization_parameter)
        self.assertEqual(np.shape(expected), np.shape(observed))
        np.testing.assert_allclose(observed, expected)
        # for e, o in zip(expected,
        #                 observed):
        #     try:
        #         self.assertAlmostEqual(e, o)
        #     except TypeError:
        #         raise AssertionError(repr(o))

    def testLeaveOneOutErrorsGivenConstantVectorInputAndNoRegularisation(self):
        self.checkLeaveOneOut([[0], [0]],
                              {'zero': 1,
                               'one': 1,
                               },
                              0)

    def testLeaveOneOutErrorsGivenConstantVectorInputAndRegularisation(self):
        expected = [[-0.5], [-0.5]]
        self.checkLeaveOneOut(expected,
                              {'zero': 1,
                               'one': 1,
                              },
                              1)

    def testLeaveOneOutErrorsGivenVariableVectorInputAndNoRegularisation(self):
        self.checkLeaveOneOut([[-1], [1]],
                              {'zero': 2,
                               'one': 1,
                               },
                              0)


    def testLeaveOneOutErrorsGivenVariableMatrixInputAndNoRegularisation(self):
        self.checkLeaveOneOut([[-1, -2],
                               [1, 2]],
                              {'zero': [2, 4],
                               'one': [1, 2],
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



class GivenTwoNodesAndTwoLinearFieldComponents(_GivenTwoNodesBase,
                                               _GivenMappingAsMeasurements):
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
                                    list(expected['func'])),
                                {'zero': 1, 'one': 2},
                                regularization_parameter=0.5)


class GivenTwoNodesAndThreeLinearFieldComponents(_GivenTwoNodesBase,
                                                 _GivenMappingAsMeasurements):
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


if __name__ == '__main__':
    unittest.main()
