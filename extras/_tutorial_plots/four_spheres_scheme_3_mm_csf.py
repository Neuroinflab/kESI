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


def plot_arrow(ax, r, color, angle, label):
    angle_rad = np.pi * angle / 180
    _X = r * np.array([[0.00, -0.05,],
                       [0.00,  0.00,],
                       [0.00,  0.05,]])
    _Y = r * np.array([[0.00,  0.90,],
                       [0.50,  1.00,],
                       [1.00,  0.90,]])
    ax.plot(np.cos(angle_rad) * _X - np.sin(angle_rad) * _Y,
            np.sin(angle_rad) * _X + np.cos(angle_rad) * _Y,
            color=color)
    plt.text(np.sign(angle) * 7.5 * np.cos(angle_rad) - 0.5 * r * np.sin(angle_rad),
             np.sign(angle) * 7.5 * np.sin(angle_rad) + 0.5 * r * np.cos(angle_rad),
             label,
             rotation=angle - (90 if angle > 0 else 270),
             color=color,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)

def plot_layer(radius,
               color,
               label,
               angle,
               r_label,
               angle_label,
               ls):
    angle_rad = np.pi * angle / 180
    plot_circle(ax, radius,
                facecolor='none',
                edgecolor=color,
                linewidth=2,
                label=label,
                ls=ls)
    plot_arrow(ax, radius - 2, color,
               angle,
               r_label)
    plt.text(-np.sin(angle_label * np.pi / 180) * (radius - 7.5),
             np.cos(angle_label * np.pi / 180) * (radius - 7.5),
             label,
             rotation=angle_label,
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)

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


    plot_layer(90,
               cbf.VERMILION,
               'SCALP',
               60,
               '90 mm',
               40,
               '-')

    plot_layer(75,
               cbf.ORANGE,
               'SKULL',
               30,
               '86 mm',
               10,
               '--')

    plot_layer(60,
               cbf.BLUE,
               'CBF',
               -45,
               '82 mm',
               -20,
               '-.')

    label = 'BRAIN'
    radius = 45
    color = cbf.GREEN
    plot_circle(ax, radius,
                facecolor='none',
                edgecolor=color,
                linewidth=2,
                label=label,
                ls=':')
    plot_arrow(ax, radius - 2, color,
               -120,
               '79 mm')
    plt.text(0,
             -20,
             label,
             color=cbf.BLACK,
             fontsize='large',
             clip_on=False,
             horizontalalignment='center',
             verticalalignment='center',
             linespacing=1.5)


plt.show()
