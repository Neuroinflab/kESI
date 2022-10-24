import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import cbf as cbf



H = 0.3
bottom = -H * 0.309
left = -H * 0.618 / 3
right = H * (1 + 0.618 * 2 / 3)
top = 1.309 * H


plt.xkcd()
for seed in [36]:
    fig = plt.figure()
    np.random.seed(seed)
    for x in np.linspace(left + bottom, right, 9):
        a, b = np.random.normal(scale=0.02 * H, size=2)
        c = np.random.normal(scale=0.01 * H)
        d = np.random.normal(scale=0.1 * H)
        plt.plot([x + a, x + b - bottom - np.abs(d)],
                 [-0.05 * H -np.abs(c), bottom + np.abs(d)],
                 color=cbf.ORANGE)

    for y in np.linspace(H,
                         top - 0.03 * H, 7, endpoint=False)[2:]:
        a, b = np.random.normal(scale=0.02 * H, size=2)
        c = 0.6 * H + np.random.normal(scale=0.1 * H)
        x = left + np.random.random() * (right - left - c)
        plt.plot([x, x + c],
                 [y + a, y + b],
                 color=cbf.SKY_BLUE)


    for X, Y, ls, c in [
                 ([0, 0], [0, H], ':', cbf.BLACK),
                 ([H, H], [0, H], ':', cbf.BLACK),
                 ([left, right], [H, H], '-', cbf.BLUE),
                 ([left, right], [0, 0], '-', cbf.ORANGE),
                 ]:
        plt.plot(X, Y, color=c, ls=ls)

    plt.xticks([])
    plt.yticks([])

    plt.plot(H * (np.array([[0.00, -0.05, -0.05],
                            [0.00,  0.00,  0.00],
                            [0.00,  0.05,  0.05]])
                  - 0.618 / 6),
             H * np.array([[0.02, 0.90,  0.10],
                           [0.50, 0.97,  0.02],
                           [0.97, 0.90,  0.10]]),
             color=cbf.BLACK)
    plt.text(-H * 0.618 / 4,
             0.5 * H,
             '0.3 mm',
             rotation=90,
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)

    plt.plot(H * np.array([[0.02, 0.90,  0.10],
                           [0.50, 0.98,  0.02],
                           [0.98, 0.90,  0.10]]),
             H * (np.array([[0.00, -0.05, -0.05],
                            [0.00,  0.00,  0.00],
                            [0.00,  0.05,  0.05]])
                  + 0.618 / 6),
             color=cbf.BLACK)
    plt.text(0.5 * H,
             H * 0.618 / 4,
             '0.3 mm',
             rotation=0,
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)

    plt.text(1.15 * H, 0.7 * bottom,
             'GLASS',
             color=cbf.ORANGE,
             fontsize='large',
             clip_on=False)
    plt.text(1.1 * H, 0.5 * H,
             'tiSSUe',
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False)
    plt.text(1.12 * H, 0.72 * H + 0.28 * top,
             'SALINE',
             color=cbf.BLUE,
             fontsize='large',
             clip_on=False)
    plt.text(0.5 * H, 0.618 * H,
             # 'hic\nsunt\ndracones',
             'region\nof\ninterest',
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)
    plt.xlim(left, right)
    plt.ylim(bottom, top)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

plt.show()
