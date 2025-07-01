import argparse
import os.path

import nibabel
import numpy as np
import pylab as pb
import pyvista
from nibabel.affines import apply_affine
from pyvista import StructuredGrid


def read_nifti(file, frame_number=0):
    nifti_img = nibabel.load(file)
    data = nifti_img.get_fdata()
    # only support single volumes

    if len(data.shape) == 5:
        try:
            data = data.get_fdata()[:, :, :, :, frame_number]
        except IndexError:
            data = data.get_fdata()[:, :, :, frame_number, :]

    data = np.squeeze(data)
    assert len(data.shape) == 3
    affine_mm = nifti_img.affine

    scale = 1 / 1000  # mm to meters

    # Scale entire 3x3 block (including skew)
    R_m = affine_mm[:3, :3] * scale

    # Scale translation vector
    T_m = affine_mm[:3, 3] * scale

    # Construct new affine in meters
    affine_m = np.eye(4)
    affine_m[:3, :3] = R_m
    affine_m[:3, 3] = T_m

    nx, ny, nz = data.shape

    i, j, k = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij"
    )
    ijk = np.vstack([i.ravel(), j.ravel(), k.ravel()]).T  # shape: (N, 3)

    xyz = apply_affine(affine_m, ijk)

    # Create structured grid
    grid = StructuredGrid()
    grid.points = xyz
    grid.dimensions = (nx, ny, nz)
    grid["values"] = data.ravel(order="C")
    # slices = grid.slice_orthogonal()
    # slices.plot(show_bounds=True, show_axes=True)

    return grid


def read_volume(file, nifti_frame_number=0):
    if file.lower().endswith("nii.gz"):
        return read_nifti(file, nifti_frame_number)
    else:
        return pyvista.read(file)


def main():
    parser = argparse.ArgumentParser(
        description="Draw crossection nifty files, works well only with no skew or rotation, rectilinear affine")
    parser.add_argument("files", nargs='+', type=str,
                        help="nifty or vtk files. Sampled nifti files are read well, but not MRI, some weird bug")
    parser.add_argument("-s", type=float, help="Line start, mm, x y z", nargs=3)
    parser.add_argument("-e", type=float, help="Line end, mm, x y z", nargs=3)
    parser.add_argument("-dx", type=float, help="sampling resolution in mm", default=1)
    parser.add_argument("-a", "--attribute", type=str, help="attribute name to draw")
    parser.add_argument("-f", "--frame_number", type=int, help="frame/component number", default=0)
    parser.add_argument("-m", "--mri", type=str, help="MRI file to use as a background, has a weird bug...", default=None)
    parser.add_argument("-g", type=float,
                        help="position on the line to use as common reference, in mm, by default none", default=None)
    parser.add_argument("-r", type=float, nargs='+', help="draw vertical lines", default=None)
    parser.add_argument("-n", type=str, help="normalize y/n", default="n")
    parser.add_argument("-l", "--labels", type=str, nargs='+', help="plot line labels per file", default=None)
    parser.add_argument("-u", "--units", type=str, help="Y axis label. defaults to potential", default="Potential [V]")

    args = parser.parse_args()

    fig = pb.figure()

    for nr, file in enumerate(args.files):
        if args.labels is not None:
            name = args.labels[nr]
        else:
            name = "{} {}".format(os.path.basename(os.path.dirname(file)), os.path.basename(file))

        volume = read_volume(file, args.frame_number)

        line_start = np.array(args.s) / 1000
        line_end = np.array(args.e) / 1000

        length = np.linalg.norm(line_end - line_start)

        resolution = length / (args.dx / 1000)

        line = pyvista.Line(line_start, line_end, int(resolution))

        sampled_volume = line.sample(volume)

        points = sampled_volume.points
        data_x = np.linalg.norm(points - points[0], axis=1) * 1000  # show in mm
        try:
            scalars = sampled_volume[args.attribute]  # replace with your scalar name
        except KeyError:
            scalars = sampled_volume['values']
        data_slice = scalars

        if args.g is not None:
            ref_level_id = np.argmin(np.abs(data_x - args.g))
            ref_level = data_slice[ref_level_id]
            pb.plot(data_x, data_slice - ref_level, label=name)
        elif args.n == 'y':
            normalized = data_slice - np.nanmin(data_slice)
            normalized = normalized / np.nanmax(normalized)
            pb.plot(data_x, normalized, label=name)

        else:
            pb.plot(data_x, data_slice, label=name)
    if args.r:
        for r in args.r:
            pb.axvline(r, linestyle='--', color='black')

    if args.mri is not None:
        direction = line_start - line_end
        origin = (line_start + line_end) / 2
        # Choose an arbitrary vector not parallel to the direction
        arbitrary = np.array([0, 0, 1])  # e.g., Z-axis
        if np.allclose(np.cross(direction, arbitrary), 0):
            arbitrary = np.array([0, 1, 0])  # fallback
        normal = np.cross(direction, arbitrary)

        mri_volume = read_volume(args.mri)

        plane = pyvista.Plane(origin,
                              direction=normal,
                              i_size=length / 2,
                              j_size=length / 2,
                              i_resolution=int(resolution),
                              j_resolution=int(resolution),
                              )
        sampled = plane.sample(mri_volume)
        scalars_1d = sampled['values']
        nx = int(resolution) + 1
        ny = int(resolution) + 1
        scalars_2d = scalars_1d.reshape(ny, nx)

        line_ax = pb.gca()
        line_ax.set_facecolor('none')
        ax_img = line_ax.twinx()
        ax_img.set_zorder(line_ax.get_zorder() - 1)
        ax_img.get_yaxis().set_visible(False)
        ax_img.imshow(scalars_2d, cmap='viridis', aspect='auto', origin='lower', zorder=-10,
                      extent=[0, length * 1000, -length / 2 * 1000, length / 2 * 1000])
        pb.sca(line_ax)

    pb.xlabel("Position along sampling line {} - {} [mm]".format(line_start * 1000, line_end * 1000))
    pb.ylabel(args.units)
    pb.legend()
    pb.show()


if __name__ == '__main__':
    main()
