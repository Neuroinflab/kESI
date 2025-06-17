import argparse
import os.path

import numpy as np
import pyvista

from kesi.fem_utils.pyvista_resampling import pyvista_sample_points
import pylab as pb

def main():
    parser = argparse.ArgumentParser(description="Draw crossection of electrode potentials")
    parser.add_argument("vtk", nargs='+', type=str, help="Mesh files (in meter)")
    parser.add_argument("attribute", type=str, help="attribute name to draw")
    parser.add_argument("-z", type=float, help="Slice at Z coordinate in mm, if default - will use electrode coordinates", default=0)
    parser.add_argument("-y", type=float, help="Slice at Y coordinate mm, if default - will use electrode coordinates", default=0)
    parser.add_argument("-dx", type=float, help="sampling resolution in mm", default=1)
    parser.add_argument("-g", type=float, help="position on X axis to use as common reference, by default none", default=None)
    parser.add_argument("-r", type=float, nargs='+', help="draw vertical lines", default=None)
    parser.add_argument("-l", "--labels", type=str, nargs='+', help="plot line labels per file", default=None)
    parser.add_argument("-n", type=str,  help="normalize y/n", default="n")

    args = parser.parse_args()

    fig = pb.figure()

    for i, mesh_file in enumerate(args.vtk):
        mesh = pyvista.read(mesh_file)
        x_min, x_max = mesh.bounds[0:2]
        x_min, x_max = -1, 1

        sampling_x = np.arange(x_min, x_max, args.dx / 1000)
        sampling_points = np.vstack([sampling_x,
                                     np.ones_like(sampling_x) * args.y / 1000,
                                     np.ones_like(sampling_x) * args.z / 1000,
                                     ]).T

        sampled_mesh = pyvista_sample_points(mesh, sampling_points)

        values =  sampled_mesh.get_array(args.attribute)
        positions = sampled_mesh.points[:, 0]

        vtk_name = os.path.basename(mesh_file)
        dirname = os.path.basename(os.path.dirname(mesh_file))
        if args.labels is not None:
            label = args.labels[i]
        else:
            label = "{} {} {}".format(dirname, vtk_name, args.attribute)

        if args.g is not None:
            ref_level_id = np.argmin(np.abs(positions - args.g / 1000))
            ref_level = values[ref_level_id]
            pb.plot(positions * 1000, values-ref_level, label=label)
        elif args.n == 'y':
            normalized = values - np.nanmin(values)
            normalized = normalized / np.nanmax(normalized)
            pb.plot(positions * 1000, normalized, label=label)
        else:
            pb.plot(positions * 1000, values, label=label)

    if args.r:
        for r in args.r:
            pb.axvline(r, linestyle='--', color='black')
    pb.legend()
    pb.xlabel("Position in X [mm]")
    pb.ylabel("Potential [V]")
    pb.show()


if __name__ == '__main__':
    main()