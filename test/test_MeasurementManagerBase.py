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
    from ._common import Stub
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    from _common import Stub

from kesi._engine import FunctionalFieldReconstructor


class TestMeasurementManagerBaseBase(unittest.TestCase):
    @property
    def MM_MISSING_ATTRIBUTE_ERRORS(self):
        yield 'load', 'LoadMethod'
        yield 'number_of_measurements', 'NumberOfMeasurementsAttribute'

    def setUp(self):
        if hasattr(self, 'CLASS'):
            self.manager = self.CLASS()
        else:
            self.skipTest('Test in virtual class called')

    def testHas_MeasurementManagerHasNoLoadMethodError_TypeErrorAttribute(self):
        self.checkTypeErrorAttribute('MeasurementManagerHasNoLoadMethodError')

    def testHas_MeasurementManagerHasNoNumberOfMeasurementsAttributeError_TypeErrorAttribute(self):
        self.checkTypeErrorAttribute('MeasurementManagerHasNoNumberOfMeasurementsAttributeError')

    def testLoadMethodIsIdentity(self):
        self.assertIs(self,
                      self.manager.load(self))

    def testNumberOfMeasurementsIsNone(self):
        self.assertIsNone(self.manager.number_of_measurements)

    def testWhenMeasurementManagerLacksAttributesThenRaisesException(self):
        for missing, exception in self.MM_MISSING_ATTRIBUTE_ERRORS:
            measurement_manager = self.getIncompleteMeasurementManager(missing)
            exception_name = 'MeasurementManagerHasNo{}Error'.format(exception)
            for ExceptionClass in [getattr(self.CLASS, exception_name),
                                   TypeError]:
                with self.assertRaises(ExceptionClass):
                    self.CLASS.validate(measurement_manager)

    def getIncompleteMeasurementManager(self, missing):
        return Stub(**{attr: None
                       for attr, _ in self.MM_MISSING_ATTRIBUTE_ERRORS
                       if attr != missing})

    def checkTypeErrorAttribute(self, attribute):
        self.assertTrue(hasattr(self.CLASS, attribute))
        self.assertTrue(issubclass(getattr(self.CLASS, attribute),
                                   TypeError))


class TestMeasurementManagerBase(TestMeasurementManagerBaseBase):
    CLASS = FunctionalFieldReconstructor.MeasurementManagerBase

    @property
    def MM_MISSING_ATTRIBUTE_ERRORS(self):
        for row in super(TestMeasurementManagerBase,
                         self).MM_MISSING_ATTRIBUTE_ERRORS:
            yield row

        yield 'probe', 'ProbeMethod'

    def testProbeMethodIsAbstract(self):
        with self.assertRaises(NotImplementedError):
            self.manager.probe(None)

    def testHas_MeasurementManagerHasNoProbeMethodError_TypeErrorAttribute(self):
        self.checkTypeErrorAttribute('MeasurementManagerHasNoProbeMethodError')


if __name__ == '__main__':
    unittest.main()