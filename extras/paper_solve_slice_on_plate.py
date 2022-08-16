#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser

import FEM.fem_slice_point_new as fspn
import FEM.fem_common as fc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve slice FEM solutions.')
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
    parser.add_argument('-q', '--quiet',
                        dest='quiet',
                        action='store_true',
                        help='do not print results',
                        default=False)

    args = parser.parse_args()

    setup_time = fc.fc.Stopwatch()
    total_solving_time = fc.fc.Stopwatch()
    with setup_time:
        function_manager = fc.FunctionManagerINI(args.config)
        fem = fspn.SlicePointSourcePotentialFEM(function_manager)

    name = args.name
    ex, ey, ez = [function_manager.getfloat(name, a) for a in 'xyz']
    conductivity = fem.base_conductivity(ex, ey, ez)

    if not args.quiet:
        print(' solving')

    with total_solving_time:
        potential_corr = fem.correction_potential(ex, ey, ez)

    metadata = {'filename': function_manager.get(name, 'filename'),
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
                }
    function_manager.store(name, potential_corr, metadata)

    config = configparser.ConfigParser()
    config.add_section(name)
    for k, v in metadata.items():
        config.set(name, k, v if isinstance(v, str) else repr(v))
    config.write(open(args.output, 'w'))
