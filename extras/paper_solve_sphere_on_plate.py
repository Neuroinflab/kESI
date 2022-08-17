#!/usr/bin/env python
# coding: utf-8

import argparse

import FEM.fem_sphere_point_new as fspn
import FEM.fem_common as fc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve sphere on plate FEM solutions.')
    parser.add_argument('-o', '--output',
                        metavar='<metadata.ini>',
                        dest='output',
                        help='path to the metadata file',
                        nargs='?')
    parser.add_argument('-c', '--config',
                        metavar='<config.ini>',
                        dest='config',
                        help='path to the FEM config file',
                        nargs='?')
    parser.add_argument('-n', '--name',
                        metavar='<solution name>',
                        dest='name',
                        help='name of the solution',
                        nargs='?')
    parser.add_argument('-g', '--grounded_plate_edge_z',
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
    config = fc.LegacyConfigParser(args.config)
    fem_mesh = config.getpath('fem', 'mesh')
    fem_degree = config.getint('fem', 'degree')
    fem_element_type = config.get('fem', 'element_type')
    model_config = config.getpath('fem', 'config')
    model_grounded_plate_edge_z = args.grounded_plate_edge_z

    name = args.name
    function_filename = config.function_filename(name)

    setup_time = fc.fc.Stopwatch()
    total_solving_time = fc.fc.Stopwatch()

    with fc.MetadataStorage(args.output,
                            ['fem',
                             'model',
                             'electrode',
                             'correction']) as metadata:
        metadata.setpath('fem', 'mesh', fem_mesh)
        metadata.set('fem', 'degree', fem_degree)
        metadata.set('fem', 'element_type', fem_element_type)
        metadata.setpath('model', 'config', model_config)
        metadata.set('model', 'grounded_plate_edge_z', model_grounded_plate_edge_z)

        with setup_time:
            function_manager = fc.FunctionManager(fem_mesh,
                                                  fem_degree,
                                                  fem_element_type)
            fem = fspn.SphereOnGroundedPlatePointSourcePotentialFEM(
                               function_manager,
                               model_config,
                               grounded_plate_edge_z=model_grounded_plate_edge_z)
            metadata.set('correction',
                         'global_preprocessing_time',
                         float(fem.global_preprocessing_time))

        metadata.set('correction', 'setup_time', float(setup_time))

        electrode_coords = [config.getfloat(name, a) for a in 'xyz']
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
