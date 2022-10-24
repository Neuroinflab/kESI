import matplotlib.pyplot as plt

def plot_circle(ax, r, **kwargs):
    ax.add_artist(plt.Circle((0, 0), r, **kwargs))

import cbf
import numpy as np
_T = np.linspace(-np.pi, np.pi)
h_grounded_plate = 88

def plot_circle2(ax, radius, color, **kwargs):
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    ax.add_artist(plt.Circle((0, 0), radius,
                             edgecolor=color,
                              facecolor=colorsys.hls_to_rgb(h, l * 1.5, s),
                             **kwargs))


def plot_arrow(ax, r, color, angle=0):
    _X = r * np.array([[0.00, -0.05, -0.05],
                       [0.00,  0.00,  0.00],
                       [0.00,  0.05,  0.05]])
    _Y = r * np.array([[0.00, 0.90,  0.10],
                       [0.50, 1.00,  0.00],
                       [1.00, 0.90,  0.10]])
    ax.plot(np.cos(angle) * _X - np.sin(angle) * _Y,
            np.sin(angle) * _X + np.cos(angle) * _Y,
            color=color)
            

with plt.xkcd():
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')

    ax.set_xlim(-110, 110)
    ax.set_ylim(-120, 100)

    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    
    _amp = np.sqrt(90 ** 2 - h_grounded_plate ** 2)
    _T = np.linspace(-_amp, _amp, 1025)
    _X = list(_T)
    _Y = list(-np.sqrt(90 ** 2 - np.square(_T)) - 3)
    _gnd_x = _X[-1]
    _X.append(_gnd_x)
    _Y.append(-110)
    ax.plot(_X, _Y,
            color=cbf.BLACK)
    ax.plot([_gnd_x - 10, _gnd_x + 10], [-100, -100],
            color=cbf.BLACK)
    ax.plot([_gnd_x - 6, _gnd_x + 6], [-104, -104],
            color=cbf.BLACK)
    ax.plot([_gnd_x - 3, _gnd_x + 3], [-107, -107],
            color=cbf.BLACK)
    plot_circle(ax, 90,
                facecolor='none',
                edgecolor=cbf.GREEN,
                linewidth=2,
                label='BRAIN')
    plot_arrow(ax, 88, cbf.GREEN,
               angle=0.25 * np.pi)
    plt.text(-25, 35,
             '90 mm',
             rotation=-45,
             color=cbf.GREEN,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)
    plt.text(10, -30,
             'BRAIN',
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)
    #plt.legend(loc='best')


plt.show()
