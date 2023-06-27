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

import numpy as np

from .. import forward_model


class SphericalForwardModel(forward_model.Spherical):
    pass


class SliceForwardModel(forward_model.Slice):
    def __init__(self, mesh, degree, config,
                 quiet=True,
                 ground_potential=0.0,
                 element_type="CG"):
        super().__init__(mesh, degree, config,
                         quiet=quiet,
                         ground_potential=ground_potential,
                         element_type=element_type)


def eigh(kernel):
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]
