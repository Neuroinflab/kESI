import argparse
import os

import numpy as np
import pandas as pd
import pyvista
from tqdm.auto import tqdm

def point_potential_solve_mesh(electrode_position, mesh, conductivity):
    distance_to_electrode = np.linalg.norm(np.array(electrode_position) - mesh.points, ord=2, axis=1)
    v_kcsd = 1.0 / (4 * np.pi * conductivity * distance_to_electrode)
    return v_kcsd


def main():
    parser = argparse.ArgumentParser(description=("Calculates an anaylytical infinite space isotropic solution "
                                                 "for point source inverse leadfields per electrode, "
                                                  "using mesh as a solution space/grid"
                                                  )
                                     )
    parser.add_argument("meshfile",
                        help=('MFEM compatible mesh, assumes it has'
                              ' one boundary condition physical group (material) and N materials of different'
                              ' conductivity, all coordinates are assumed to be in meters')
                        )
    parser.add_argument("electrodefile",
                        help=('CSV with electrode names and positions, in milimeters, with a header of: \n'
                              '\tlabel,x,y,z')
                        )
    parser.add_argument("output", type=str,
                        help=("output folder with results."
                              "Results will be saved as VTK attributes with names:"
                              "potential_ELECTRODE_NAME, and correction_ELECTRODE_NAME"
                              )
                        )

    parser.add_argument('-c', "--conductivity", type=float,
                        help=("Universe conductivity"),
                        default=0.33)

    namespace = parser.parse_args()

    electrodes = pd.read_csv(namespace.electrodefile)

    outdir = namespace.output
    output_filename = os.path.join(outdir, os.path.splitext(os.path.basename(namespace.meshfile))[0] + '.vtk')
    os.makedirs(outdir, exist_ok=True)

    mesh = pyvista.read(namespace.meshfile)

    # singlethreaded electrode sim
    results = []
    for row_id, electrode in tqdm(electrodes.iterrows(), desc="simulating electrodes", total=len(electrodes)):
        # electrodes in mm, mesh in meters
        electrode_position = electrode[["x", "y", "z"]].astype(float).values / 1000
        result = point_potential_solve_mesh(electrode_position, mesh,
                                            conductivity=namespace.conductivity)
        results.append(result)

    for result, electrode_name in tqdm(list(zip(results, electrodes['label'].values)),
                                       desc='saving output potential'):
        mesh.point_data["potential_{}".format(electrode_name)] = result

    mesh.save(output_filename)


if __name__ == '__main__':
    main()
