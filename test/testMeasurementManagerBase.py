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

from kesi._engine import FunctionalFieldReconstructor


class TestMeasurementManagerBase(unittest.TestCase):
    CLASS = FunctionalFieldReconstructor.MeasurementManagerBase

    def setUp(self):
        self.manager = self.CLASS()

    def testLoadMethodIsIdentity(self):
        self.assertIs(self,
                      self.manager.load(self))

    def testProbeMethodIsAbstract(self):
        with self.assertRaises(NotImplementedError):
            self.manager.probe(None)

    def testNumberOfMeasurementsIsNone(self):
        self.assertIsNone(self.manager.number_of_measurements)


if __name__ == '__main__':
    unittest.main()