#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import configparser

import numpy as np

import FEM.fem_slice_point_new as fspn
import FEM.fem_common as fc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample slice FEM solutions.')
    parser.add_argument('-o', '--output',
                        metavar='<output.npz>',
                        dest='output',
                        help='path to the output file',
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
    parser.add_argument('-k',
                        type=int,
                        dest='k',
                        metavar="<k>",
                        help="each dimansion is sampled in 2**k + 1 points",
                        default=9)
    parser.add_argument('-r', '--sampling_radius',
                        type=float,
                        dest='radius',
                        metavar="<radius>",
                        help='length of the sampled cube side [m]',
                        default=3e-4)
    parser.add_argument('-f', '--fill',
                        type=float,
                        dest='fill',
                        metavar="<fill value>",
                        help='fill value',
                        default=np.nan)

    args = parser.parse_args()


    N = 2 ** args.k + 1
    X = np.linspace(-args.radius / 2, args.radius / 2, N)
    Y = np.linspace(-args.radius / 2, args.radius / 2, N)
    Z = np.linspace(0, args.radius, N)

    loading_start = datetime.datetime.now()
    fem = fspn.SlicePointSourcePotentialFEM(fc.FunctionManagerINI(args.config))
    preprocess_start = datetime.datetime.now()

    try:
        print(args.name)
        correction_potential = fem._fm.load(args.name)
    except configparser.NoSectionError:
        print(' SOLUTION NOT FOUND')
        exit(-1)

    CORRECTION_POTENTIAL = np.full((len(X),
                                    len(Y),
                                    len(Z)),
                                   args.fill)

    ERROR_R = []
    start = datetime.datetime.now()
    if not args.quiet:
        print(f'{args.name}\tLOADING: {preprocess_start - loading_start}')
        print(f'{args.name}\tPREPROCESSING: {start - preprocess_start}')
    for i_x, x in enumerate(X):
        if not args.quiet:
            print(
                f'{args.name}\t{i_x}\t{x * 1000_000:.1f}um\t{datetime.datetime.now() - start}')

        for i_y, y in enumerate(Y):
            for i_z, z in enumerate(Z):
                try:
                    CORRECTION_POTENTIAL[i_x, i_y, i_z] = correction_potential(x, y, z)
                except RuntimeError:
                    ERROR_R.append((x, y, z))

    np.savez_compressed(args.output,
                        CORRECTION_POTENTIAL=CORRECTION_POTENTIAL,
                        X=X,
                        Y=Y,
                        Z=Z,
                        LOCATION=[fem._fm.getfloat(args.name, c) for c in
                                  'xyz'],
                        BASE_CONDUCTIVITY=fem._fm.getfloat(args.name,
                                                           'base_conductivity'),
                        _PREPROCESSING_TIME=(
                                start - preprocess_start).total_seconds(),
                        _PROCESSING_TIME=(
                                datetime.datetime.now() - start).total_seconds())

    if ERROR_R != []:
        print('ERROR R:')
        print(ERROR_R)