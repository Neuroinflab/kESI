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

import sys

import numpy as np

class FunctionalFieldReconstructor(object):
    def __init__(self, field_components, measurement_manager):
        self._field_components = field_components
        self._measurement_manager = measurement_manager
        self._generate_kernels()

    def _generate_kernels(self):
        self._generate_pre_kernel()
        self._generate_kernel()

    def _generate_kernel(self):
        self._kernel = np.dot(self._pre_kernel.T,
                              self._pre_kernel) * self._pre_kernel.shape[0]

    def _generate_pre_kernel(self):
        m = len(self._field_components)
        n = self._measurement_manager.number_of_measurements
        self._pre_kernel = np.empty((m, n))
        self._fill_evaluated_components(self._pre_kernel)
        self._pre_kernel /= m

    def _fill_evaluated_components(self, evaluated):
        for i, component in enumerate(self._field_components):
            evaluated[i, :] = self._measurement_manager.probe(component)

    def __call__(self, measurements, regularization_parameter=0):
        return LinearMixture(zip(self._field_components,
                                 np.dot(self._pre_kernel,
                                        self._solve_kernel(
                                              self._measurement_vector(measurements),
                                              regularization_parameter)
                                        ).flatten()))

    def _measurement_vector(self, values):
        return self._measurement_manager.load(values).reshape(-1, 1)

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


class LinearMixture(object):
    def __init__(self, components=[]):
        self._components, self._weights = [], []

        try:
            for c, w in components:
                if isinstance(c, LinearMixture):
                    self._append_components_from_mixture(self._components,
                                                         self._weights,
                                                         self._components,
                                                         c * w)
                else:
                    self._components.append(c)
                    self._weights.append(w)

        except TypeError:
            self._components = (components,)
            self._weights = (1,)

        self._prepare_cache_for_dir()

    def _prepare_cache_for_dir(self):
        components = self._components
        self._dir = ({attr for attr in dir(components[0])
                      if all(hasattr(c, attr) for c in components[1:])}
                     if components
                     else ())

    def __getattr__(self, name):
        if name not in self._dir:
            raise AttributeError

        def wrapper(*args, **kwargs):
            return sum(w * getattr(c, name)(*args, **kwargs)
                       for w, c in zip(self._weights,
                                       self._components))

        return wrapper

    def __dir__(self):
        return list(self._dir)

    def __add__(self, other):
        return self._add(other)

    def __radd__(self, other):
        return self._add(other)

    def _add(self, other):
        if other == 0:
            return self

        components = list(self._components)
        weights = list(self._weights)
        self._append_components_from_mixture(components, weights,
                                             self._components,
                                             other)

        return self.__class__(list(zip(components, weights)))

    @staticmethod
    def _append_components_from_mixture(components,
                                        weights,
                                        reference_components,
                                        mixture):
        for c, w in zip(mixture._components,
                        mixture._weights):
            try:
                i = reference_components.index(c)
            except ValueError:
                weights.append(w)
                components.append(c)
            else:
                weights[i] += w

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        return self._mul(other)

    def __rmul__(self, other):
        return self._mul(other)

    def _mul(self, other):
        if other == 1 or self == 0:
            return self

        return self.__class__([(c, w * other)
                               for c, w
                               in zip(self._components,
                                      self._weights)])

    def __truediv__(self, other):
        return self._div(other)

    def _div(self, other):
        return 1. / other * self

    def __eq__(self, other):
        if self:
            return self is other

        return not other

    def __ne__(self, other):
        if self:
            return self is not other

        return bool(other)

    def _bool(self):
        return bool(self._components)

    # Use API appropriate for current Python version
    if sys.version_info.major > 2:
        def __bool__(self):
            return self._bool()

    else:
        def __nonzero__(self):
            return self._bool()

        def __div__(self, other):
            return self._div(other)
