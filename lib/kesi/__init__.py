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
import operator

try:
    import pandas as pd

except:
    pd = None


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


class FunctionalKernelFieldReconstructor(object):
    class _FieldApproximator(object):
        def __init__(self, field_components, field_weights):
            self._weighted_components = sorted(zip(field_weights,
                                                   field_components),
                                               key=operator.itemgetter(0))

        def __getattr__(self, item):
            def f(*args, **kwargs):
                return sum(w * getattr(c, item)(*args, **kwargs)
                           for w, c in self._weighted_components)

            return f

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
        self._field_components = field_components
        self._nodes = nodes
        self._generate_kernels(input_domain)

    def _generate_kernels(self, input_domain):
        self._generate_pre_crosskernel(input_domain)
        self._generate_kernel()

    def _generate_kernel(self):
        self._kernel = np.dot(self._pre_cross_kernel.T,
                              self._pre_cross_kernel) * self._pre_cross_kernel.shape[0]

    def _generate_pre_crosskernel(self, name):
        n = len(self._field_components)
        self._pre_cross_kernel = np.empty((n,
                                           len(self._nodes)))
        self._fill_evaluated_components(self._pre_cross_kernel,
                                        name)
        self._pre_cross_kernel /= n

    def _fill_evaluated_components(self, evaluated, name):
        for i, component in enumerate(self._field_components):
            evaluated[i, :] = getattr(component, name)(self._nodes)

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
        return self._FieldApproximator(self._field_components,
                                       np.dot(self._pre_cross_kernel,
                                              self._solve_kernel(
                                                  self._measurement_vector(measurements),
                                                  regularization_parameter)
                                              ).flatten())

    def _measurement_vector(self, values):
        return np.array([values[n] for n in (self._nodes)]).reshape(-1, 1)

    def _solve_kernel(self, measurements, regularization_parameter=0):
        K = self._kernel
        return np.linalg.solve(K + np.identity(K.shape[0])
                                   * regularization_parameter,
                               measurements)

    def leave_one_out_errors(self, measured, regularization_parameter):
        n = self._kernel.shape[0]
        KERNEL = self._kernel + regularization_parameter * np.identity(n)
        IDX_N = np.arange(n)
        X = self._measurement_vector(measured)
        return [self._leave_one_out_estimate(KERNEL, X, i, IDX_N != i) - x[0]
                for i, x in enumerate(X)]

    def _leave_one_out_estimate(self, KERNEL, X, i, IDX):
        return np.dot(KERNEL[np.ix_([i], IDX)],
                      np.linalg.solve(KERNEL[np.ix_(IDX, IDX)],
                                      X[IDX, :]))[0, 0]
