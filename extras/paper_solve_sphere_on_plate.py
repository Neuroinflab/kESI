#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser

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
    config = configparser.ConfigParser()
    config.read(args.config)

    setup_time = fc.fc.Stopwatch()
    total_solving_time = fc.fc.Stopwatch()
    with setup_time:
        function_manager = fc.FunctionManagerINI(args.config)
        fem = fspn.SphereOnGroundedPlatePointSourcePotentialFEM(function_manager,
                                                                grounded_plate_edge_z=args.grounded_plate_edge_z)

    name = args.name
    ex, ey, ez = [config.getfloat(name, a) for a in 'xyz']
    conductivity = fem.base_conductivity(ex, ey, ez)

    if not args.quiet:
        print(' solving')

    with total_solving_time:
        potential_corr = fem.correction_potential(ex, ey, ez)

    metadata = {'filename': config.get(name, 'filename'),
          'x': ex,
          'y': ey,
          'z': ez,
          'setup_time': float(setup_time),
          'total_solving_time': float(total_solving_time),
          'local_preprocessing_time': float(
              fem.local_preprocessing_time),
          'global_preprocessing_time': float(
              fem.global_preprocessing_time),
          'solving_time': float(fem.solving_time),
          'base_conductivity': conductivity,
          'grounded_plate_edge_z': args.grounded_plate_edge_z,
                }
    function_manager.legacy_store(name, potential_corr, metadata)

    metadata_config = configparser.ConfigParser()
    metadata_config.add_section(name)
    for k, v in metadata.items():
        metadata_config.set(name, k, v if isinstance(v, str) else repr(v))
    metadata_config.write(open(args.output, 'w'))


    # function_manager.write(function_manager.getpath('fem', 'solution_metadata_filename'))
