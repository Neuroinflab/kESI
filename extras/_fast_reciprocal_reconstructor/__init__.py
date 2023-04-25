#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Institute of Applied Psychology;  #
#    Faculty of Management and Social Communication; Jagiellonian University) #
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

import warnings

from kesi.kernel import potential_basis_functions as pbf
from kesi.kernel.constructor import (deprecated,
                                     Convolver,
                                     KernelConstructor,
                                     CrossKernelConstructor)


# DEPRECATED


class ckESI_convolver(Convolver):
    @deprecated('class ckESI_convolver', 'Convolver class')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecated('.ds() method', '.steps()')
    def ds(self, name):
        return self.steps()

    @property
    @deprecated('.POT_MESH attribute', '.POT_GRID')
    def POT_MESH(self):
        return self.POT_GRID

    @property
    @deprecated('.CSD_MESH attribute', '.CSD_GRID')
    def CSD_MESH(self):
        return self.CSD_GRID

    @property
    @deprecated('.SRC_MESH attribute', '.SRC_GRID')
    def SRC_MESH(self):
        return self.SRC_GRID


class ckESI_kernel_constructor(KernelConstructor):
    @deprecated('class ckESI_kernel_constructor', 'KernelConstructor class')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ckESI_crosskernel_constructor(CrossKernelConstructor):
    @deprecated('class ckESI_crosskernel_constructor',
                'CrossKernelConstructor class')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @deprecated('.csd_indices attribute', '.csd_mask')
    def csd_indices(self):
        return self.csd_mask


class PAE_kCSD_Analytical(pbf.Analytical):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_Analytical class is \
deprecated, use Analytical instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kCSD_Numerical(pbf.Numerical):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_Numerical class is \
deprecated, use Numerical instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kCSD_AnalyticalMasked(pbf.AnalyticalMaskedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_AnalyticalMasked class is \
    deprecated, use AnalyticalMaskedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kCSD_NumericalMasked(pbf.NumericalMasked):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kCSD_NumericalMasked class is \
deprecated, use NumericalMasked instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kESI_AnalyticalMasked(pbf.AnalyticalMaskedAndCorrectedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kESI_AnalyticalMasked class is \
    deprecated, use AnalyticalMaskedAndCorrectedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class PAE_kESI_Analytical(pbf.AnalyticalCorrectedNumerically):
    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("PAE_kESI_Analytical class is \
    deprecated, use AnalyticalCorrectedNumerically instead"),
                      stacklevel=2)
        super().__init__(*args, **kwargs)
