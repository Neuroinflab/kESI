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

import numpy as np

try:
    import pandas as pd

except:
    pd = None


from ._engine import FunctionalFieldReconstructor, MeasurementManagerBase


class _KernelFieldApproximator(object):
    def __init__(self, kernels, cross_kernels, nodes, points,
                 regularization_parameter):
        self._kernels = kernels
        self._cross_kernels = cross_kernels
        self._nodes = nodes
        self._points = points
        self.regularization_parameter = regularization_parameter

    def copy(self, regularization_parameter=None):
        return _KernelFieldApproximator(self._kernels,
                                        self._cross_kernels,
                                        self._nodes,
                                        self._points,
                                        self.regularization_parameter
                                          if regularization_parameter is None
                                          else regularization_parameter)

    def __call__(self, field, measured_field, measurements):
        return self._wrap_results(measurements,
                                  self._points[field],
                                  self._approximate(field,
                                                    measured_field,
                                                    measurements))

    def _wrap_results(self, measurements, keys, values):
        if pd and isinstance(measurements, pd.Series):
            return pd.Series(data=values,
                             index=keys)
        return dict(zip(keys, values))

    def _approximate(self, field, measured_field, measurements):
        return np.dot(self._cross_kernels[measured_field, field],
                      self._solve_kernel(measured_field,
                                         self._measurement_vector(measured_field,
                                                                  measurements))).flatten()

    def _solve_kernel(self, field, measurements):
        K = self._kernels[field]
        return np.linalg.solve(K + np.identity(K.shape[0])
                                   * self.regularization_parameter,
                               measurements)

    def _measurement_vector(self, name, values):
        nodes = self._nodes[name]
        rhs = np.array([values[n] for n in nodes]).reshape(-1, 1)
        return rhs


class KernelFieldApproximator(_KernelFieldApproximator):
    class _KernelGenerator(object):
        def __init__(self, field_components, nodes, points):
            self._components = {name: [getattr(f, name)
                                       for f in field_components]
                                for name in set(nodes) | set(points)}
            self.nodes = {k: v if isinstance(v, np.ndarray) else list(v)
                          for k, v in nodes.items()}
            self.points = {k: v if isinstance(v, np.ndarray) else list(v)
                           for k, v in points.items()}

        def _make_kernel(self, name):
            NDS = self._evaluate_components(name, self.nodes[name])
            return np.dot(NDS.T, NDS)

        def _make_cross_kernel(self, src, dst):
            PTS = self._evaluate_components(dst, self.points[dst])
            NDS = self._evaluate_components(src, self.nodes[src])
            return np.dot(PTS.T, NDS)

        def _evaluate_components(self, name, points):
            components = self._components[name]
            evaluated = np.empty((len(components), len(points)))
            for i, f in enumerate(components):
                evaluated[i, :] = f(points)

            return evaluated

        @property
        def kernels(self):
            return {name: self._make_kernel(name)
                    for name in self.nodes}

        @property
        def cross_kernels(self):
            return {(src, dst): self._make_cross_kernel(src, dst)
                    for src in self.nodes
                    for dst in self.points
                    }

    def __init__(self, field_components, nodes, points,
                 regularization_parameter=0):
        generator = self._KernelGenerator(field_components,
                                          nodes,
                                          points)
        super(KernelFieldApproximator,
              self).__init__(generator.kernels,
                             generator.cross_kernels,
                             {k: self._ndarrays_to_tuples(v)
                              for k, v in generator.nodes.items()
                              },
                             {k: self._ndarrays_to_tuples(v)
                              for k, v in generator.points.items()
                              },
                             regularization_parameter)

    def _ndarrays_to_tuples(self, point):
        if isinstance(point, np.ndarray):
            return tuple(map(self._ndarrays_to_tuples, point))

        return point


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
