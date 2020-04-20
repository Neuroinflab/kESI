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
from io import BytesIO

try:
    from . import (test_MeasurementManagerBase,
                   test_Engine,
                   test_Verbose,
                   test_LoadableFunctionalFieldReconstructor)
    from ._common import TestCase
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import test_MeasurementManagerBase
    import test_Engine
    import test_Verbose
    import test_LoadableFunctionalFieldReconstructor
    from _common import TestCase

from kesi._engine import (FunctionalFieldReconstructor,
                          LoadableFunctionalFieldReconstructor)
from kesi._verbose import LoadableVerboseFFR


class _PlainMatrixMM(test_Verbose._CommonPlainMatrixMM,
                     LoadableVerboseFFR.MeasurementManagerBase):
    pass


class _TrueBasesMatrixMM(test_Verbose._CommonTrueBasesMatrixMM,
                         _PlainMatrixMM):
    pass


class TestKernelMatricesOfVerboseFFR(test_Verbose._CommonTestKernelMatricesOfVerboseFFR):
    CLASS = LoadableVerboseFFR

    def getReconstructor(self, measurements_of_basis_functions):
        self.measurement_mgr = self.MM(measurements_of_basis_functions)
        measurement_manager = self.measurement_mgr
        reconstructor = FunctionalFieldReconstructor(measurement_manager.bases(),
                                                     measurement_manager)
        buffer = BytesIO()
        reconstructor.save(buffer)
        buffer.seek(0)
        return self.CLASS(buffer,
                          measurement_manager.bases(),
                          measurement_manager)

    def testIsSubclassOfFunctionalFieldReconstructor(self):
        self.assertTrue(issubclass(self.CLASS,
                                   LoadableFunctionalFieldReconstructor))


class TestKernelFunctionsOfVerboseFFR(test_Verbose._CommonTestKernelFunctionsOfVerboseFFR,
                                      TestKernelMatricesOfVerboseFFR):
    pass


class MockTestKernelFunctionsOfVerboseFFR(test_Verbose._CommonMockTestKernelFunctionsOfVerboseFFR,
                                          TestKernelFunctionsOfVerboseFFR):
    pass


class TestsOfInitializationErrors(test_LoadableFunctionalFieldReconstructor.TestsOfInitializationErrors):
    CLASS = LoadableVerboseFFR

    MM_MISSING_ATTRIBUTE_ERRORS = (
            test_LoadableFunctionalFieldReconstructor.TestsOfInitializationErrors.MM_MISSING_ATTRIBUTE_ERRORS
            + [('probe_at_single_point', 'ProbeAtSinglePointMethod')])


class TestMeasurementManagerBase(test_Verbose.TestVerboseMeasurementManagerBase):
    CLASS = LoadableVerboseFFR.MeasurementManagerBase


if __name__ == '__main__':
    unittest.main()
