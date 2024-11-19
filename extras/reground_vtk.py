import argparse

import numpy as np
import pyvista

from kesi.fem_utils.pyvista_resampling import pyvista_sample_points


def main():
    parser = argparse.ArgumentParser(description="Draw crossection nifty files, works well only with no skew or rotation, rectilinear affine")
    parser.add_argument("file", type=str, help="path to vtk")
    parser.add_argument("output", type=str, help="path to vtk_regrounded")
    # between FCz and Cz in 10-20 on a sphere at R 0.88
    parser.add_argument("-x", type=float, help="Ground value position in meters", default=0.000333449764541583)
    parser.add_argument("-y", type=float, help="Ground value position in meters", default=0.0420608102042992)
    parser.add_argument("-z", type=float, help="Ground value position in meters", default=0.0750644933029997)

    args = parser.parse_args()

    mesh = pyvista.read(args.file)
    new_mesh = mesh.copy()
    new_mesh.clear_point_data()

    points = np.array([[args.x, args.y, args.z]])

    sampled = pyvista_sample_points(mesh, points)

    potential_names = [i for i in mesh.point_data.keys() if i.startswith('potential')]

    for electrode in potential_names:
        leadfield = mesh.point_data[electrode]
        grounding_value = sampled.point_data[electrode][0]
        regrounded = leadfield-grounding_value
        new_mesh.point_data[electrode] = regrounded
    new_mesh.save(args.output)


if __name__ == '__main__':
    main()