#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

from ._engine import (FunctionalFieldReconstructor,
                      LoadableFunctionalFieldReconstructor,
                      MeasurementManagerBase,
                      _LinearKernelSolver)
from ._verbose import _CrossKernelReconstructor

from . import kernel
from . import common


class FunctionalKernelFieldReconstructor(FunctionalFieldReconstructor):
    class _MeasurementManager(MeasurementManagerBase):
        def __init__(self, name, nodes):
            self._nodes = nodes
            self._name = name
            self.number_of_measurements = len(nodes)

        def probe(self, field):
            return getattr(field, self._name)(self._nodes)

        def load(self, measured):
            return [measured[k] for k in self._nodes]

    def __init__(self, field_components, input_domain, nodes):
        """
        :param field_components: assumed components of the field [#f1]_
        :type field_components: Sequence(Component)

        :param input_domain: the scalar quantity of the field the interpolation
                             is based on [#f1]_
        :type input_domain: str

        :param nodes: estimation points of the ``input_domain`` [#f1]_
        :type nodes: Sequence(key)

        .. rubric:: Footnotes

        .. [#f1] ``Component`` class objects are required to have a method which
                 name is given as ``input_domain``. The method called with
                 ``nodes`` as  its only argument is required to return
                 a sequence of values of the ``input_domain`` quantity for
                 the component.
        """
        super(FunctionalKernelFieldReconstructor,
              self).__init__(field_components,
                             self._MeasurementManager(input_domain,
                                                      nodes))

    def __call__(self, measurements, regularization_parameter=0):
        """
        :param measurements: values of the field quantity in the estimation
                             points (see the docstring of the
                             :py:meth:`constructor<__init__>` for details.
        :type measurements: Mapping(key, float)

        :param regularization_parameter: the regularization parameter
        :type regularization_parameter: float

        :return: interpolator of field quantities
        :rtype: an object implementing methods of the same names and signatures
                as those of ``Component`` objects (provided as the argument
                ``field_components`` of the :py:meth:`constructor<__init__>`.
        """
        return super(FunctionalKernelFieldReconstructor,
                     self).__call__(measurements,
                                    regularization_parameter=regularization_parameter)


class Reconstructor(_CrossKernelReconstructor):
    """
    A wrapper around (cross)kernel matrices facilitating
    CSD reconstruction without explicit matrix operations.
    """
    def __init__(self, kernel, crosskernel):
        """
        Parameters
        ----------
        kernel : ndarray
            N x N element kernel matrix

        crosskernel : ndarray
            C x N element cross-kernel matrix (or 3+D array if C is a tuple);
            for C see docstring of the `.__call__()` method
        """
        super().__init__(_LinearKernelSolver(kernel),
                         crosskernel)

    def __call__(self, potential, regularization_parameter=0):
        """
        Parameters
        ----------
        potential : ndarray
            N [x T] element vector (or matrix) of measured potential values;
            N is fixed (see docstring of the `.__init__()` method)

        regularization_parameter : float or int, optional
            regularization parameter

        Returns
        -------
        ndarray
            C [x T] element vector (or 2+D array); C is fixed and may be a tuple
            (see docstring of the `.__init__()` method)
        """
        return super().__call__(potential, regularization_parameter)
