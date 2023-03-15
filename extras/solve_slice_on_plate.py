#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2021 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
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

import argparse
import configparser

import FEM.fem_slice_point_new as fspn
import FEM.fem_common as fc


if __name__ == "__main__":
    def get_coords(file, electrode):
        try:
            config = configparser.ConfigParser()
            config.read(file)
            return [config.getfloat(electrode, a) for a in "xyz"]
        except:
            import pandas as pd
            DF = pd.read_csv(file,
                             index_col="NAME")
            return DF.loc[electrode, ["X", "Y", "Z"]]


    parser = argparse.ArgumentParser(description="Solve slice FEM solutions.")
    parser.add_argument("-o", "--output",
                        metavar="<metadata.ini>",
                        dest="output",
                        required=True,
                        help="path to the metadata file")
    parser.add_argument("-c", "--config",
                        metavar="<config.ini>",
                        dest="config",
                        required=True,
                        help="path to the model config file")
    parser.add_argument("-e", "--electrodes",
                        metavar="<electrodes.ini>",
                        dest="electrodes",
                        required=True,
                        help="path to the electrode location config file")
    parser.add_argument("-n", "--name",
                        metavar="<electrode name>",
                        dest="name",
                        required=True,
                        help="name of the electrode")
    parser.add_argument("-m", "--mesh",
                        metavar="<mesh.xdmf>",
                        dest="mesh",
                        required=True,
                        help="path to the FEM mesh")
    parser.add_argument("-d", "--degree",
                        type=int,
                        metavar="<FEM element degree>",
                        dest="degree",
                        help="degree of FEM elements",
                        default=1)
    parser.add_argument("--element-type",
                        metavar="<FEM element type>",
                        dest="element_type",
                        help="type of FEM elements",
                        default="CG")
    parser.add_argument("-g", "--ground-potential",
                        type=float,
                        dest="ground_potential",
                        metavar="<ground potential>",
                        help="the potential at the grounded slice-covering dome")
    parser.add_argument("-q", "--quiet",
                        dest="quiet",
                        action="store_true",
                        help="do not print results",
                        default=False)

    args = parser.parse_args()

    function_filename = args.output[:-3] + "h5"

    setup_time = fc.fc.Stopwatch()
    total_solving_time = fc.fc.Stopwatch()

    with fc.MetadataStorage(args.output,
                            ["fem",
                             "model",
                             "electrode",
                             "correction"]) as metadata:
        metadata.setpath("fem", "mesh", args.mesh)
        metadata.set("fem", "degree", args.degree)
        metadata.set("fem", "element_type", args.element_type)
        metadata.setpath("model", "config", args.config)

        with setup_time:
            function_manager = fc.FunctionManager(args.mesh,
                                                  args.degree,
                                                  args.element_type)
            fem = fspn.SlicePointSourcePotentialFEM(function_manager,
                                                    args.config,
                                                    ground_potential=args.ground_potential)

            metadata.set("correction",
                         "global_preprocessing_time",
                         float(fem.global_preprocessing_time))

        metadata.set("correction", "setup_time", float(setup_time))

        electrode_coords = get_coords(args.electrodes, args.name)
        for k, v in zip("xyz", electrode_coords):
            metadata.set("electrode", k, v)

        if not args.quiet:
            print(" solving")

        with total_solving_time:
            potential_corr = fem.correction_potential(*electrode_coords)

        metadata.setfields(
            "correction",
            {
                "total_solving_time": float(total_solving_time),
                "local_preprocessing_time": float(fem.local_preprocessing_time),
                "solving_time": float(fem.solving_time),
                "base_conductivity": fem.base_conductivity(*electrode_coords),
            })

        function_manager.store(function_filename,
                               potential_corr)
        metadata.setpath("correction", "filename", function_filename)
