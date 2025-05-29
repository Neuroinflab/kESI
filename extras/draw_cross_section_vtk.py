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
    parser.add_argument("-x", type=float, help="Slice at X coordinate in mm, if default - will use electrode coordinates", default=0)
    parser.add_argument("-y", type=float, help="Slice at Y coordinate mm, if default - will use electrode coordinates", default=0)
    parser.add_argument("-dx", type=float, help="sampling resolution in mm", default=1)
    parser.add_argument("-g", type=float, help="position on Z axis to use as common reference, by default none", default=None)

    args = parser.parse_args()

    fig = pb.figure()

    for mesh_file in args.vtk:
        mesh = pyvista.read(mesh_file)
        z_min, z_max = mesh.bounds[-2:]

        sampling_z = np.arange(z_min, z_max, args.dx / 1000)
        sampling_points = np.vstack([np.ones_like(sampling_z) * args.x / 1000,
                                     np.ones_like(sampling_z) * args.y / 1000,
                                     sampling_z]).T

        sampled_mesh = pyvista_sample_points(mesh, sampling_points)

        values =  sampled_mesh.get_array(args.attribute)
        positions = sampled_mesh.points[:, 2]

        vtk_name = os.path.basename(mesh_file)
        dirname = os.path.basename(os.path.dirname(mesh_file))
        label = "{} {} {}".format(dirname, vtk_name, args.attribute)

        if args.g is not None:
            ref_level_id = np.argmin(np.abs(positions - args.g / 1000))
            ref_level = values[ref_level_id]
            pb.plot(positions, values-ref_level, label=label)
        else:
            pb.plot(positions, values, label=label)

    # testing stuff
    distance_to_source = np.abs(positions - 0.0232)
    conductivity = 1.5
    amplitude_ampers = 1
    full_anal = 1.0 / (4 * np.pi * conductivity * distance_to_source) * amplitude_ampers
    pb.plot(positions, full_anal, label="fully analytical")
    ###

    pb.legend()
    pb.show()


if __name__ == '__main__':
    main()