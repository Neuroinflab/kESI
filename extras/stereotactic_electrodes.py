#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021 Jakub M. Dzik (Jagiellonian University)               #
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

import configparser
import os.path

import numpy as np

class ElectrodesBrainLabACPC(object):
    def __init__(self, filename):
        self._config = configparser.ConfigParser()
        self._config.read(filename)
        self._root_directory = os.path.dirname(os.path.abspath(filename))
        self._electrodes = {name: self._load_electrode(name)
                            for name in self._config.sections()}

    def _load_electrode(self, name):
        x, y, z = [self._config.getfloat(name, c)
                   for c in 'xyz']
        angle1, angle2 = [self._config.getfloat(name, f'angle{i}')
                          for i in (1, 2)]
        config = self._getpath(name, 'config')
        color = self._config.get(name, 'color',
                                 fallback=None)
        return self._BrainLabElectrode(name, x, y, z,
                                       angle1 * np.pi / 180,
                                       angle2 * np.pi / 180,
                                       config,
                                       color)

    def _getpath(self, section, field):
        path = self._config.get(section, field)
        if os.path.isabs(path):
            return path

        return os.path.join(self._root_directory,
                            path)

    class _BrainLabElectrode(object):
        def __init__(self, name, x, y, z, angle1, angle2,
                     config, color):
            self._config = configparser.ConfigParser()
            self._config.read(config)

            self.name = name
            self.target = x, y, z
            self.color = color

            length = self._config.getfloat('_properties', 'length')

            cos1, sin1 = np.cos(angle1), np.sin(angle1)
            # ele1 = [sin1 * length, 0, cos1 * length]
            # electrode parallel to Z-axis rotates about Y axis by angle1 first

            if abs(angle1) > np.pi / 2:  # just guessing for angle1 < -90
                cos2, sin2 = np.cos(angle2 + np.pi), np.sin(angle2 + np.pi)
            else:
                cos2, sin2 = np.cos(angle2), np.sin(angle2)

            # rotate about X axis
            self._entry_vector = np.array([sin1, sin2 * cos1, cos2 * cos1])

            self.entry = self.target + self._entry_vector * length

            self._contacts = {c: self._load_contact(c)
                              for c in self._config.sections()
                              if not c.startswith('_')}

        @property
        def target(self):
            return np.array([self.target_x, self.target_y, self.target_z])

        @target.setter
        def target(self, value):
            self.target_x, self.target_y, self.target_z = value

        @property
        def entry(self):
            return np.array([self.entry_x, self.entry_y, self.entry_z])

        @entry.setter
        def entry(self, value):
            self.entry_x, self.entry_y, self.entry_z = value

        def _load_contact(self, name):
            distance = self._config.getfloat(name, 'distance')
            x, y, z = self.target + self._entry_vector * distance
            return self._Contact(x, y, z)

        class _Contact(object):
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
