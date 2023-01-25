#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create 4SM kernels.')
    parser.add_argument('-o', '--output',
                        metavar='<output.npz>',
                        dest='output',
                        help='output prefix')
    parser.add_argument('input',
                        metavar='<source prefix>',
                        nargs=2,
                        help='prefixes of eigensources to be combined')

    args = parser.parse_args()

    with np.load(f'{args.input[0]}_analysis.npz') as fh:
        ES_1T = fh['EIGENSOURCES'].T

    with np.load(f'{args.input[1]}_analysis.npz') as fh:
        PROJECTION = np.matmul(ES_1T, fh['EIGENSOURCES'])
        del ES_1T

    BEST_MATCH = np.argmax(abs(PROJECTION), axis=1)
    SIGN = np.sign(PROJECTION[range(len(BEST_MATCH)),
                              BEST_MATCH]).reshape(1, 1, 1, -1)

    with np.load(f'{args.input[1]}_eigensources.npz') as fh:
        CSD = 0.5 * SIGN * fh['CSD'].take(BEST_MATCH, axis=-1)

    with np.load(f'{args.input[0]}_eigensources.npz') as fh:
        kwargs = {k: fh[k] for k in 'XYZ'}
        CSD += 0.5 * fh['CSD']

    np.savez_compressed(args.output,
                        CSD=CSD,
                        **kwargs)