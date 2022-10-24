import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import cbf as cbf



H = 0.3
bottom = -H * 0.309
left = -H * 0.618 / 3
right = H * (1 + 0.618 * 2 / 3)
top = 1.309 * H

RESOLUTION = 32
XX = H * np.linspace(1.0 / RESOLUTION,
                     (1 - 1.0 / RESOLUTION),
                     RESOLUTION - 1)
XX, YY = np.meshgrid(XX,
                     [round(x, 10) for x in XX])
IDX_X, IDX_Y = np.meshgrid(range(RESOLUTION-1),
                           range(RESOLUTION-1))
ele_coords = np.array([XX.flatten(),
                       YY.flatten()])
ele_names = [f'E{i[1]:03d}_{i[0]:03d}' for i in zip(IDX_X.flatten(),
                                                    IDX_Y.flatten())]


ELECTRODES = pd.DataFrame(ele_coords.T,
                          columns=['X', 'Y'],
                          index=ele_names)
electrode_names = [f'E{3 + i * 3:03d}_{3 + j * 6:03d}'
                       for i in range(8)
                       for j in range(5)]
RECORDING_ELECTRODES = ELECTRODES.loc[electrode_names][['X', 'Y']]

plt.xkcd()
for seed in [36]:# [1]:
    fig = plt.figure()
    # plt.title(str(seed))
    np.random.seed(seed) #0
    plt.plot([left, right],
             [0, 0],
             color=cbf.ORANGE)
    for x in np.linspace(left + bottom, right, 9):
        a, b = np.random.normal(scale=0.02 * H, size=2)
        c = np.random.normal(scale=0.01 * H)
        d = np.random.normal(scale=0.1 * H)
        plt.plot([x + a, x + b - bottom - np.abs(d)],
                 [-0.05 * H -np.abs(c), bottom + np.abs(d)],
                 color=cbf.ORANGE)

    # plt.plot([-H, 2*H],
    #          [H, H],
    #          color=cbf.SKY_BLUE)
    for y in np.linspace(H,
                         top - 0.03 * H, 7, endpoint=False)[2:]:
        a, b = np.random.normal(scale=0.02 * H, size=2)
        c = 0.6 * H + np.random.normal(scale=0.1 * H)
        x = left + np.random.random() * (right - left - c)
        plt.plot([x, x + c],
                 [y + a, y + b],
                 color=cbf.SKY_BLUE)

    #plt.scatter(RECORDING_ELECTRODES.X,
    #            RECORDING_ELECTRODES.Y,
    #            marker='x',
    #            color=cbf.BLACK)

    # for x, y in zip(RECORDING_ELECTRODES.X, RECORDING_ELECTRODES.Y):
    #     s = H / 20
    #     a, b, c, d = np.random.normal(scale=0.1 * s, size=4)
    #     plt.plot([x - s + a, x + s - a],
    #              [y - s + b, y + s - b],
    #              color=cbf.BLACK)
    #     plt.plot([x - s + c, x + s - c],
    #              [y + s + d, y - s - d],
    #              color=cbf.BLACK)

    # plt.plot([0, H, H, 0, 0],
    #          [0, 0, 2 * H, 2*H, 0.01 * H],
    #          color=cbf.BLACK,
    #          ls=':')
    # plt.plot([0.01 * H, 0.98 * H],
    #          [H, H],
    #          color=cbf.BLACK,
    #          ls=':')


    for X, Y, ls, c in [
                 ([left, right], [H, H], '-', cbf.BLUE),
                 # ([0, H], [H, H], '-', cbf.BLACK),
                 ([0, 0], [0, H], ':', cbf.BLACK),
                 ([H, H], [0, H], ':', cbf.BLACK),
                 # ([left, right], [0, 0]),
                 # ([left, H], [2*H, 2*H]),
                 # ([left, H], [2 * H, 2 * H]),
                 # ([0, 0], [1.99*H, bottom]),
                 # ([H, H], [1.98*H, bottom]),
                 # ([0, 0], [1.99*H, bottom]),
                 # ([H, H], [1.98*H, bottom]),
                 ]:
        plt.plot(X, Y, color=c, ls=ls)

    plt.xticks([0, H],
               ['-0.15 mm', '0.15 mm'])
    plt.yticks([0, H, 2*H],
               ['0\nmm', '0.3\nmm', '0.6\nmm'])

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
    plt.text(0.5 * H, 0.55 * H,
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
