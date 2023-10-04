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


def one_hot_vector(length, hot_position, hot=1, cold=0):
    return np.where(np.arange(length) == hot_position, hot, cold)


def shape(dimensions, axis):
    return one_hot_vector(dimensions, axis, hot=-1, cold=1)


def reshape(A, axis, n=3):
    return np.reshape(A, shape(n, axis))


if __name__ == '__main__':

    # TESTS one_hot_vector()

    assert np.all(one_hot_vector(1, 0) == [1])
    assert np.all(one_hot_vector(2, 0) == [1, 0])
    assert np.all(one_hot_vector(2, 1) == [0, 1])
    assert np.all(one_hot_vector(2, 1, hot=-1) == [0, -1])
    assert np.all(one_hot_vector(2, 1, hot=-1, cold=1) == [1, -1])