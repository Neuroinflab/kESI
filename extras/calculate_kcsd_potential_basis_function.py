#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

import pandas as pd

from kesi.kernel import potential_basis_functions as pbf
from kesi.kernel.electrode import Conductivity as Electrode

from _calculate_potential_basis_function import ScriptBase


class Script(ScriptBase):
    PotentialBasisFunction = pbf.Analytical

    class ArgumentParser(ScriptBase.ArgumentParser):
        def __init__(self):
            super().__init__("kCSD")
            self.add_argument("-e", "--electrodes",
                              required=True,
                              metavar="<electrodes.csv>",
                              help="locations of electrodes")
            self.add_argument("--conductivity",
                              type=float,
                              default=0.33,
                              metavar="<conductivity [S/m]>",
                              help="medium conductivity")

    def _load_electrodes(self, args):
        ELES = pd.read_csv(args.electrodes,
                           index_col="NAME",
                           usecols=["NAME", "X", "Y", "Z"]).loc[args.names]
        self._electrodes = {name: Electrode(x, y, z, args.conductivity)
                            for name, (x, y, z) in ELES.iterrows()}

    def run(self):
        self._run(self._electrodes)

    def _get_convolver_grid(self):
        return self.CENTROID_XYZ

    def _get_quadrature(self):
        return []

    def _get_src_mask(self, convolver):
        return self.CENTROID_MASK


if __name__ == "__main__":
    model = Script()
    model.run()
