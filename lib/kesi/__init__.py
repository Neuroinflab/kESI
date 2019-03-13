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

class KernelFieldInterpolator(object):
    def __init__(self, fieldComponents, nodes, points, lambda_=0):
        self._components = {name: [getattr(f, name)
                                   for f in fieldComponents]
                            for name in set(nodes) | set(points)}
        self._nodes = {k: list(v) for k, v in nodes.items()}
        self._points = {k: list(v) for k, v in points.items()}

        self._K = {name: self._makeKernel(name)
                   for name in self._nodes}
        self._crossK = {(src, dst): self._makeCrossKernel(src, dst)
                        for src in self._nodes
                        for dst in self._points
                        }
        self._invK = {name: np.linalg.inv(K + np.eye(*K.shape) * lambda_)
                      for name, K in self._K.items()
                      }

    def _makeKernel(self, name):
        NDS = self._evaluateComponents(name, self._nodes[name])
        return np.dot(NDS, NDS.T)

    def _makeCrossKernel(self, src, dst):
        PTS = self._evaluateComponents(dst, self._points[dst])
        NDS = self._evaluateComponents(src, self._nodes[src])
        return np.dot(PTS, NDS.T)

    def _evaluateComponents(self, name, points):
        components = self._components[name]
        return np.array([[f(p) for f in components]
                         for p in points])

    def __call__(self, field, measuredField, measurements):
        nodes = self._nodes[measuredField]
        rhs = np.array([measurements[n] for n in nodes]).reshape(-1, 1)
        invK = self._invK[measuredField]
        values = np.dot(self._crossK[measuredField, field],
                        np.dot(invK, rhs)).flatten()
        keys = self._points[field]

        if pd and isinstance(measurements, pd.Series):
            return pd.Series(data=values,
                             index=keys)
        return dict(zip(keys, values))
