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

class KernelFieldInterpolator(object):
    def __init__(self, fieldComponents, nodes, points):
        self._nodes = {k: list(v) for k, v in nodes.items()}
        self._points = {k: list(v) for k, v in points.items()}

        self._K = {name: np.matrix([[sum(getattr(f, name)(a) * getattr(f, name)(b)
                                         for f in fieldComponents)
                                     for b in nds
                                     ]
                                    for a in nds])
                   for name, nds in self._nodes.items()}
        self._crossK = {(src, dst): np.matrix([[sum(getattr(f, src)(n) * getattr(f, dst)(p)
                                                    for f in
                                                    fieldComponents)
                                                for n in nds
                                                ]
                                                for p in pts
                                               ])
                        for src, nds in self._nodes.items()
                        for dst, pts in self._points.items()
                        }

    def __call__(self, field, measuredField, measurements):
        nodes = self._nodes[measuredField]
        rhs = np.matrix([measurements[n] for n in nodes]).T
        invK = np.linalg.inv(self._K[measuredField])
        values = np.array(self._crossK[measuredField, field] * invK * rhs).flatten()
        return dict(zip(self._points[field], values))
