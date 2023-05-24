#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#    Copyright (C) 2021-2023 Jakub M. Dzik (Institute of Applied Psychology;  #
#    Faculty of Management and Social Communication; Jagiellonian University) #
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

from itertools import repeat

import numpy as np
from matplotlib import pyplot as plt

from .. import forward_model
from .. import cbf


class SphericalForwardModel(forward_model.Spherical):
    pass


class SliceForwardModel(forward_model.Slice):
    def __init__(self, mesh, degree, config,
                 quiet=True,
                 ground_potential=0.0,
                 element_type="CG"):
        super().__init__(mesh, degree, config,
                         quiet=quiet,
                         ground_potential=ground_potential,
                         element_type=element_type)


class coordinate_planes_CoordinatePlanesVisualisation(object):
    def __init__(self,
                 grid,
                 plane_intersection,
                 dpi=35,
                 cmap=cbf.bwr,
                 amp=None,
                 length_factor=1,
                 length_unit='$m$',
                 unit_factor=1,
                 unit=''):
        self.grid = grid
        self.plane_intersection = np.array(plane_intersection)
        self.dpi = dpi
        self.cmap = cmap
        self.amp = amp
        self.length_factor = length_factor
        self.length_unit = length_unit
        self.unit_factor = unit_factor
        self.unit = unit

    def plot_planes(self,
                    DATA_PLANES,
                    title=None,
                    amp=None):
        DATA_ZY, DATA_ZX, DATA_XY = self._parse_planes(*DATA_PLANES)

        wx, wy = DATA_XY.shape
        wz = DATA_ZY.shape[0]
        assert DATA_ZY.shape[1] == wy
        assert DATA_ZX.shape == (wz, wx)

        self.start_new_image(title, wx, wy, wz)
        self._plot_planes([DATA_ZY, DATA_ZX, DATA_XY],
                          amp if amp is not None else self._max(*DATA_PLANES))
        self.finish_image()

    def _max(self, *arrays):
        return max(abs(_A).max() for _A in arrays)

    def start_new_image(self, title, wx, wy, wz):
        self.fig = plt.figure(figsize=((wx + wy) / self.dpi,
                                       (wz + wy) / self.dpi))
        if title is not None:
            self.fig.suptitle(title)

        gs = plt.GridSpec(2, 2,
                          figure=self.fig,
                          width_ratios=[wx, wy],
                          height_ratios=[wz, wy])

        self.ax_xz = self.fig.add_subplot(gs[0, 0])
        self.ax_xz.set_aspect('equal')
        self.ax_xz.set_ylabel(f'Z [{self.length_unit}]')
        self.ax_xz.set_xlabel(f'X [{self.length_unit}]')

        self.ax_yx = self.fig.add_subplot(gs[1, 1])
        self.ax_yx.set_aspect('equal')
        self.ax_yx.set_ylabel(f'X [{self.length_unit}]')
        self.ax_yx.set_xlabel(f'Y [{self.length_unit}]')

        self.ax_yz = self.fig.add_subplot(gs[0, 1],
                                          sharey=self.ax_xz,
                                          sharex=self.ax_yx)
        self.ax_yz.set_aspect('equal')

        self.cax = self.fig.add_subplot(gs[1, 0])
        self.cax.set_visible(False)

    def _plot_planes(self, DATA_PLANES, amp):
        DATA_ZY, DATA_ZX, DATA_XY = [A * self.unit_factor for A in DATA_PLANES]

        def _extent(first, second):
            _first = self.grid[first] * self.length_factor
            _second = self.grid[second] * self.length_factor
            return (_first.min(), _first.max(),
                    _second.min(), _second.max())

        self.ax_xz.imshow(DATA_ZX,
                          vmin=-amp * self.unit_factor,
                          vmax=amp * self.unit_factor,
                          cmap=self.cmap,
                          origin='lower',
                          extent=_extent(0, 2))
        self.ax_yx.imshow(DATA_XY,
                          vmin=-amp * self.unit_factor,
                          vmax=amp * self.unit_factor,
                          cmap=self.cmap,
                          origin='lower',
                          extent=_extent(1, 0))
        self.im = self.ax_yz.imshow(DATA_ZY,
                                    vmin=-amp * self.unit_factor,
                                    vmax=amp * self.unit_factor,
                                    cmap=self.cmap,
                                    origin='lower',
                                    extent=_extent(1, 2))

    def finish_image(self):
        x, y, z = self.length_factor * self.plane_intersection

        self.ax_xz.axvline(x, ls=':', color=cbf.BLACK)
        self.ax_xz.axhline(z, ls=':', color=cbf.BLACK)

        self.ax_yx.axvline(y, ls=':', color=cbf.BLACK)
        self.ax_yx.axhline(x, ls=':', color=cbf.BLACK)

        self.ax_yz.axvline(y, ls=':', color=cbf.BLACK)
        self.ax_yz.axhline(z, ls=':', color=cbf.BLACK)
        self.fig.colorbar(self.im, ax=self.cax,
                          orientation='horizontal',
                          label=self.unit)

    def _parse_planes(self, YZ, XZ, XY):
        return YZ.T, XZ.T, XY


class slice_CoordinatePlanesVisualisation(
                               coordinate_planes_CoordinatePlanesVisualisation):
    def __init__(self,
                 grid,
                 plane_intersection,
                 dpi=35,
                 cmap=cbf.bwr,
                 amp=None,
                 length_factor=1,
                 length_unit='$m$',
                 unit_factor=1,
                 unit=''):
        super().__init__(grid,
                         plane_intersection,
                         dpi=dpi,
                         cmap=cmap,
                         amp=amp,
                         length_factor=length_factor,
                         length_unit=length_unit,
                         unit_factor=unit_factor,
                         unit=unit)
        self.indices = [np.searchsorted(g, a)
                        for a, g in zip(plane_intersection,
                                        grid)]

    def _parse_planes(self, YZ, XZ, XY, indices=repeat(0)):
        return super()._parse_planes(*self._slice_planes([YZ, XZ, XY],
                                                         indices))

    def _slice_planes(self, PLANES, indices):
        return [P[tuple(ix if i == j else slice(None) for j in range(3))]
                for i, (P, ix) in enumerate(zip(PLANES, indices))]

    def plot_volume(self, DATA, title=None, amp=None):
        self.start_new_image(title, *DATA.shape)
        self._plot_planes(self._parse_planes(DATA,
                                             DATA,
                                             DATA,
                                             self.indices),
                           amp if amp is not None else abs(DATA).max())
        self.finish_image()

    def compare_with_gt(self, GT, CSD, title=''):
        ERROR = CSD - GT
        error_L2 = np.sqrt(np.square(ERROR).sum() / np.square(GT).sum())
        amp = max(abs(CSD).max(),
                  abs(GT).max(),
                  abs(ERROR).max())
        self._plot_gt(GT, amp)
        self._plot_reconstruction(CSD, amp, title)
        self._plot_error(ERROR, amp, title, error_L2)

    def _plot_gt(self, GT, amp):
        self.plot_volume(GT,
                         title='GT CSD',
                         amp=amp)

    def _plot_reconstruction(self, CSD, amp, title):
        self.plot_volume(CSD,
                         title=f'{title} reconstruction',
                         amp=amp)

    def _plot_error(self, ERROR, amp, title, error_L2):
        self.plot_volume(ERROR,
                         title=f'{title} error (GT normalized L2 norm: {error_L2:.2g})',
                         amp=amp)

    @property
    def PLANES_XYZ(self):
        return [[[c] if i == j else A for j, A in enumerate(self.grid)]
                for i, c in enumerate(self.plane_intersection)]


class _CoordinatePlanesVisualisationSphericalBase(
    slice_CoordinatePlanesVisualisation):
    def _plot_circle(self, ax, r):
        ax.add_artist(plt.Circle((0, 0), r * self.length_factor,
                                 facecolor='none',
                                 edgecolor=cbf.BLACK,
                                 linestyle=':'))

    def _add_spheres(self, sphere_radii):
        for c, ax in zip(self.plane_intersection,
                         [self.ax_yz,
                          self.ax_xz,
                          self.ax_yx]):
            for r2 in np.square(sphere_radii):
                self._plot_circle(ax, np.sqrt(r2 - np.square(c)))


class _CoordinatePlanesVisualisationSpherical(
    _CoordinatePlanesVisualisationSphericalBase):
    def finish_image(self):
        super().finish_image()
        self._add_spheres(self.SPHERE_RADII)


class four_spheres_CoordinatePlanesVisualisation(
                                       _CoordinatePlanesVisualisationSpherical):
    SPHERE_RADII = [0.079, 0.080, 0.085, 0.090]


class single_sphere_CoordinatePlanesVisualisation(
                                       _CoordinatePlanesVisualisationSpherical):
    SPHERE_RADII = [0.090]


class case_study_CoordinatePlanesVisualisation(
                                   _CoordinatePlanesVisualisationSphericalBase):
    SPHERE_RADII = [0.079, 0.082, 0.086, 0.090]
    SPHERE_RADII_GT = [0.079, 0.080, 0.085, 0.090]

    def __init__(self,
                 grid,
                 plane_intersection,
                 dpi=17,
                 cmap=cbf.bwr,
                 amp=None,
                 length_factor=1,
                 length_unit='$m$',
                 unit_factor=1,
                 unit=''):
        super().__init__(grid,
                         plane_intersection,
                         dpi=dpi,
                         cmap=cmap,
                         amp=amp,
                         length_factor=length_factor,
                         length_unit=length_unit,
                         unit_factor=unit_factor,
                         unit=unit)

    def _plot_gt(self, GT, amp):
        super()._plot_gt(GT, amp)
        self._add_spheres(self.SPHERE_RADII_GT)

    def _plot_reconstruction(self, CSD, amp, title):
        super()._plot_reconstruction(CSD, amp, title)
        self._add_spheres(self.SPHERE_RADII)
