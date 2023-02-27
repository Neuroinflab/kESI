#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser

import FEM.fem_sphere_point_new as fspn
import FEM.fem_common as fc


if __name__ == '__main__':
    def get_coords(file, electrode):
        try:
            config = configparser.ConfigParser()
            config.read(file)
            return [config.getfloat(electrode, a) for a in 'xyz']
        except:
            import pandas as pd
            DF = pd.read_csv(file,
                             index_col='NAME')
            return DF.loc[electrode, ['X', 'Y', 'Z']]


    parser = argparse.ArgumentParser(description='Solve sphere on plate FEM solutions.')
    parser.add_argument('-o', '--output',
                        metavar='<metadata.ini>',
                        dest='output',
                        required=True,
                        help='path to the metadata file')
    parser.add_argument('-c', '--config',
                        metavar='<config.ini>',
                        dest='config',
                        required=True,
                        help='path to the model config file')
    parser.add_argument('-e', '--electrodes',
                        metavar='<electrodes.csv>',
                        dest='electrodes',
                        required=True,
                        help='path to the electrode location config file')
    parser.add_argument('-n', '--name',
                        metavar='<electrode name>',
                        dest='name',
                        required=True,
                        help='name of the electrode')
    parser.add_argument('-m', '--mesh',
                        metavar='<mesh.xdmf>',
                        dest='mesh',
                        required=True,
                        help='path to the FEM mesh')
    parser.add_argument('-d', '--degree',
                        type=int,
                        metavar='<FEM element degree>',
                        dest='degree',
                        help='degree of FEM elements',
                        default=1)
    parser.add_argument('--element-type',
                        metavar='<FEM element type>',
                        dest='element_type',
                        help='type of FEM elements',
                        default='CG')
    parser.add_argument('-g', '--grounded-plate-edge-z',
                        type=float,
                        dest='grounded_plate_edge_z',
                        metavar="<grounded plate edge's z>",
                        help='Z coordinate of the grounded plate',
                        default=-0.088)
    parser.add_argument('-q', '--quiet',
                        dest='quiet',
                        action='store_true',
                        help='do not print results',
                        default=False)

    args = parser.parse_args()

    function_filename = args.output[:-3] + 'h5'
    model_grounded_plate_edge_z = args.grounded_plate_edge_z

    setup_time = fc.fc.Stopwatch()
    total_solving_time = fc.fc.Stopwatch()

    with fc.MetadataStorage(args.output,
                            ['fem',
                             'model',
                             'electrode',
                             'correction']) as metadata:
        metadata.setpath('fem', 'mesh', args.mesh)
        metadata.set('fem', 'degree', args.degree)
        metadata.set('fem', 'element_type', args.element_type)
        metadata.setpath('model', 'config', args.config)
        metadata.set('model', 'grounded_plate_edge_z', model_grounded_plate_edge_z)

        with setup_time:
            function_manager = fc.FunctionManager(args.mesh,
                                                  args.degree,
                                                  args.element_type)
            fem = fspn.SphereOnGroundedPlatePointSourcePotentialFEM(
                               function_manager,
                               args.config,
                               grounded_plate_edge_z=model_grounded_plate_edge_z)
            metadata.set('correction',
                         'global_preprocessing_time',
                         float(fem.global_preprocessing_time))

        metadata.set('correction', 'setup_time', float(setup_time))

        electrode_coords = get_coords(args.electrodes, args.name)
        for k, v in zip('xyz', electrode_coords):
            metadata.set('electrode', k, v)

        if not args.quiet:
            print(' solving')

        with total_solving_time:
            potential_corr = fem.correction_potential(*electrode_coords)

        metadata.setfields(
            'correction',
            {
                'total_solving_time': float(total_solving_time),
                'local_preprocessing_time': float(fem.local_preprocessing_time),
                'solving_time': float(fem.solving_time),
                'base_conductivity': fem.base_conductivity(*electrode_coords),
            })

        function_manager.store(function_filename,
                               potential_corr)
        metadata.setpath('correction', 'filename', function_filename)
