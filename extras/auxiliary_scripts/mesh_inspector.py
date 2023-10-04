#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################

import sys
from argparse import ArgumentParser

from xml.dom import minidom
import collections
import operator
import itertools
import gc

try:
    import matplotlib.pyplot as plt

except ImportError:
    plt = None


class Vertex(collections.namedtuple('VertexBase', ['x', 'y', 'z', 'edges'])):
    __slots__ = ()

    def __new__(cls, *args):
        return super(Vertex, cls).__new__(cls, *(args if len(args) == 3 else args[0][:3]), edges=[])

    def __repr__(self):
        return 'Vertex({0.x}, {0.y}, {0.z}, {{<edges>}})'.format(self)

    def distance(self, other):
        return sum((a - b)**2 for a, b in zip(self[:3], other[:3])) ** 0.5


parser = ArgumentParser()
parser.add_argument('meshes',
                    nargs='+',
                    help='paths to XML FeniCS mesh files to be inspected')
parser.add_argument('--start',  '-s', '--from',
                    type=float,
                    nargs=3,
                    default=(0.0, 0.0, 0.0))
parser.add_argument('--end',  '-e', '--to',
                    type=float,
                    nargs=3)
parser.add_argument('--aggressive', '-a',
                    action='store_true',
                    help='enables aggresive memory saving policy')

if plt is not None:
    parser.add_argument('--plot', '--plt', '-p',
                        action='store_true',
                        help='plot the traversed edges')


args = parser.parse_args()
start = Vertex(args.start)

for filename in args.meshes:
    print(filename)
    xmldoc = minidom.parse(filename)
    vertices = xmldoc.getElementsByTagName('vertex')
    n_vertices = len(vertices)

    v_start = Vertex(min(((float(v.getAttribute('x')),
                           float(v.getAttribute('y')),
                           float(v.getAttribute('z')),
                           )
                          for v in vertices),
                         key=start.distance))
    end = None if args.end is None else Vertex(args.end)
    v_end = Vertex((min if end else max)(((float(v.getAttribute('x')),
                                           float(v.getAttribute('y')),
                                           float(v.getAttribute('z')),
                                           )
                                          for v in vertices),
                                         key=(end if end else start).distance))

    r_max = v_start.distance(v_end)
    vert = [None] * n_vertices

    for v in vertices:
        loc = (float(v.getAttribute('x')),
               float(v.getAttribute('y')),
               float(v.getAttribute('z')))
        if v_start.distance(loc) > r_max or v_end.distance(loc) > r_max:
           continue

        i = int(v.getAttribute('index'))
        assert vert[i] is None

        vert[i] = Vertex(loc) if loc != v_start[:3] else v_start

    #assert vert.count(None) == 0
    del vertices
    gc.collect()

    cells = xmldoc.getElementsByTagName('tetrahedron')
    del xmldoc

    title = f'{filename} ({n_vertices} vertices, {len(cells)} elements)'
    print(title)


    for c in cells:
        vs = [vert[int(c.getAttribute('v{:d}'.format(i)))] for i in range(4)]
        vs = [v for v in vs if v is not None]
        for a, b in itertools.permutations(vs, 2):
            if b not in a.edges:
                if args.aggressive \
                  and v_start.distance(b) <= v_start.distance(a) \
                  and v_end.distance(b) >= v_end.distance(a):
                    continue

                a.edges.append(b)

    del cells
    del vert
    gc.collect()

    print(f'R\tstep\tx\ty\tz')
    print(f'0\t\t{start.x:.2g}\t{start.y:.2g}\t{start.z:.2g}')
    last = start
    v = v_start
    vs = [start]
    rs = [0]
    steps = [0]
    while True:
        steps.append(last.distance(v))
        rs.append(start.distance(v))
        vs.append(Vertex(v[:3]))
        print(f'{rs[-1]:.2g}\t{steps[-1]:.2g}\t{v.x:.2g}\t{v.y:.2g}\t{v.z:.2g}')

        if v == v_end:
            break
        if not v.edges:
            assert args.aggressive
            print('Unable to continue; try to disable -a flag')
            break

        last = Vertex(v)
        v = min(v.edges, key=v_end.distance)

    if end:
        print(f'{start.distance(end):.2g}\t{v.distance(end):.2g}\t{end.x:.2g}\t{end.y:.2g}\t{end.z:.2g}')

    del v_start, v, v_end, last
    gc.collect()

    print()

    if plt and args.plot:
        plt.figure()
        plt.title(title)
        plt.plot(rs, steps)

if plt and args.plot:
    plt.show()
