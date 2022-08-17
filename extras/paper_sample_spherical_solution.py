#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime

import numpy as np

import FEM.fem_common as fc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample spherical FEM solutions.')
    parser.add_argument('-o', '--output',
                        metavar='<output.npz>',
                        dest='output',
                        help='path to the output file')
    parser.add_argument('-c', '--config',
                        metavar='<config.ini>',
                        dest='config',
                        help='path to the solution metadata')
    parser.add_argument('-q', '--quiet',
                        dest='quiet',
                        action='store_true',
                        help='do not print results',
                        default=False)
    parser.add_argument('-k',
                        type=int,
                        dest='k',
                        metavar="<k>",
                        help="each dimension is sampled in 2**k + 1 points",
                        default=9)
    parser.add_argument('-r', '--sampling-radius',
                        type=float,
                        dest='radius',
                        metavar="<radius>",
                        help='radius of the sampled sphere [m]',
                        default=0.079)
    parser.add_argument('-f', '--fill',
                        type=float,
                        dest='fill',
                        metavar="<fill value>",
                        help='fill value',
                        default=np.nan)

    args = parser.parse_args()

    N = 2 ** args.k + 1
    X = np.linspace(-args.radius, args.radius, N)
    Y = np.linspace(-args.radius, args.radius, N)
    Z = np.linspace(-args.radius, args.radius, N)

    r2_max = args.radius ** 2

    config = fc.MetadataReader(args.config)

    preprocess_start = datetime.datetime.now()
    function_manager = fc.FunctionManager(config.getpath('fem', 'mesh'),
                                          config.getint('fem', 'degree'),
                                          config.get('fem', 'element_type'))
    loading_start = datetime.datetime.now()

    correction_potential = function_manager.load(config.getpath('correction',
                                                                'filename'))

    ERROR_R = []
    start = datetime.datetime.now()

    CORRECTION_POTENTIAL = np.full((len(X),
                                    len(Y),
                                    len(Z)),
                                   args.fill)

    if not args.quiet:
        print(f'PREPROCESSING: {loading_start - preprocess_start}')
        print(f'LOADING: {start - loading_start}')
    for i_x, x in enumerate(X):
        if not args.quiet:
            print(
                f'{i_x}\t{x * 1000:.1f}mm\t{datetime.datetime.now() - start}')
        r2_x = x ** 2
        if r2_x > r2_max:
            continue

        for i_y, y in enumerate(Y):
            r2_xy = r2_x + y ** 2
            if r2_xy > r2_max:
                continue

            for i_z, z in enumerate(Z):
                if r2_xy + z ** 2 <= r2_max:
                    try:
                        CORRECTION_POTENTIAL[i_x, i_y, i_z] = correction_potential(x, y, z)
                    except RuntimeError:
                        ERROR_R.append(np.sqrt(r2_xy + z ** 2))

    np.savez_compressed(args.output,
                        CORRECTION_POTENTIAL=CORRECTION_POTENTIAL,
                        X=X,
                        Y=Y,
                        Z=Z,
                        LOCATION=[config.getfloat('electrode', c) for c in
                                  'xyz'],
                        BASE_CONDUCTIVITY=config.getfloat('correction',
                                                          'base_conductivity'),
                        _R_LIMIT=[0, args.radius],
                        _PREPROCESSING_TIME=(
                                loading_start - preprocess_start).total_seconds(),
                        _LOADING_TIME=(
                                start - loading_start).total_seconds(),
                        _PROCESSING_TIME=(
                                datetime.datetime.now() - start).total_seconds())

    if ERROR_R != []:
        print('ERROR R:')
        print(ERROR_R)