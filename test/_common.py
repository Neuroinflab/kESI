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


class Stub(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FunctionFieldComponent(Stub):
    def __init__(self, func, fprime):
        super(FunctionFieldComponent,
              self).__init__(func=self._vectorize(func),
                             fprime=self._vectorize(fprime))

    @staticmethod
    def _vectorize(f):
        def wrapper(arg):
            if isinstance(arg, list):
                return list(map(f, arg))

            return f(arg)

        return wrapper
