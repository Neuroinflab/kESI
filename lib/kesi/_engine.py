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

class FunctionalFieldReconstructor(object):
    class _FieldApproximator(object):
        def __init__(self, field_components, field_weights):
            self._components = field_components
            self._weights = field_weights

        def __getattr__(self, item):
            def f(*args, **kwargs):
                return sum(w * getattr(c, item)(*args, **kwargs)
                           for w, c in zip(self._weights,
                                           self._components))

            return f

    def __init__(self, field_components, measurement_manager):
        self._field_components = field_components
        self._measurement_nodes = measurement_manager
        self._generate_kernels()

    def _generate_kernels(self):
        self._generate_pre_crosskernel()
        self._generate_kernel()

    def _generate_kernel(self):
        self._kernel = np.dot(self._pre_cross_kernel.T,
                              self._pre_cross_kernel) * self._pre_cross_kernel.shape[0]

    def _generate_pre_crosskernel(self):
        n = len(self._field_components)
        self._pre_cross_kernel = np.empty((n,
                                           len(self._measurement_nodes)))
        self._fill_evaluated_components(self._pre_cross_kernel)
        self._pre_cross_kernel /= n

    def _fill_evaluated_components(self, evaluated):
        for i, component in enumerate(self._field_components):
            evaluated[i, :] = self._measurement_nodes.evaluate_component(component)

    def __call__(self, measurements, regularization_parameter=0):
        return self._FieldApproximator(self._field_components,
                                       np.dot(self._pre_cross_kernel,
                                              self._solve_kernel(
                                                  self._measurement_vector(measurements),
                                                  regularization_parameter)
                                              ).flatten())

    def _measurement_vector(self, values):
        return self._measurement_nodes.get_measurement_vector(values).reshape(-1, 1)

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
