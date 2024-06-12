import argparse
import os
from functools import partial
import mfem.ser as mfem
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from io import StringIO

from kesi.mfem_solver.interpolated_mfem_coefficient import CSDCoefficient


def prepare_mesh(meshfile, refinement):
    # to create run
    # gmsh -3 -format msh22 four_spheres_in_air_with_plane.geo
    print("Loading mesh...")
    mesh = mfem.Mesh(meshfile, 1, 1)
    print("Loading mesh... Done")

    if refinement:
        print("additional uniform refinement...")
        mesh.UniformRefinement()
        print("additional uniform refinement... Done")

    return mesh


def prepare_fespace(mesh):
    dim = mesh.Dimension()
    order = 1
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec)
    return fespace


def mfem_solve_mesh_multiprocessing_wrap(electrode_position, meshfile, boundary_potential, conductivities, refinement):
    try:
        device = mfem.Device("cpu")
        device.Print()
    except RuntimeError:
        pass  # already configured
    mesh = prepare_mesh(meshfile, refinement)
    coeff = electrode_coefficient(electrode_position)
    result = mfem_solve_mesh(coeff, mesh, boundary_potential, conductivities)
    sol = np.array(result.GetDataArray())
    return sol


def electrode_coefficient(electrode_position):
    # for each point charge
    point_charge_coeff = mfem.DeltaCoefficient()
    point_charge_coeff.SetScale(1)
    point_charge_coeff.SetDeltaCenter(mfem.Vector(electrode_position))
    return point_charge_coeff


def csd_distribution_coefficient(grid, values, type='nearest'):
    """grid - list of x, y, z values of grid definition - numpy arrays of grid nodes positions,
    values - 3D numpy array of CSD values
    type - "nearest" (extremely fast) or "linear" (slow) interpolation"""
    coeff = CSDCoefficient(grid[0], grid[1], grid[2], values)
    if type == 'nearest':
        coeff_func = coeff.get_nearest_neighbor_compiled_coeff()
        return coeff_func
    else:
        return coeff


def mfem_solve_mesh(csd_coefficient, mesh, boundary_potential, conductivities):
    """
    csd_coefficient - CSD distribution in coefficient form
    mesh - MFEM mesh object
    boundary_potential - value of the potential at the ground
    conductivities - numpy array of conductivities in S/m one per mesh material, can be longer than amount of materials - extra values won't not be used
    """

    fespace = prepare_fespace(mesh)
    print('Number of finite element unknowns: ' +
          str(fespace.GetTrueVSize()))

    # this is a masking list which decides, which boundaries of the mesh are Dirichlet boundaries
    # in this case we have 4 spheres, and for some reason index of the outside boundary is 5
    # for now I want to set the outside boundary of the 4 spheres as essential boundary and having a 0 potential.
    ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
    ess_bdr.Assign(0)
    ess_bdr[mesh.bdr_attributes[-1] - 1] = 1

    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    b = mfem.LinearForm(fespace)

    conductivities_vector = mfem.Vector(list(1.0 / (4 * np.pi * conductivities)))

    conductivities_coeff = mfem.PWConstCoefficient(conductivities_vector)

    b.AddDomainIntegrator(mfem.DomainLFIntegrator(csd_coefficient))
    b.Assemble()

    x = mfem.GridFunction(fespace)
    # setting initial values in all points, boundary elements will enforce this  value
    x.Assign(float(boundary_potential))

    a = mfem.BilinearForm(fespace)

    a.AddDomainIntegrator(mfem.DiffusionIntegrator(conductivities_coeff))
    a.Assemble()

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()

    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    AA = A.AsSparseMatrix()
    M = mfem.GSSmoother(AA)
    mfem.PCG(A, M, B, X, 1, 10000, 1e-12, 0.0)

    a.RecoverFEMSolution(X, b, x)

    # extract vertices and solution as numpy array
    # sol = x.GetDataArray()
    return x


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using voxel downsampling")
    parser.add_argument("meshfile",
                        help=('MFEM compatible mesh, assumes it has'
                              ' one boundary condition physical group (material) and N materials of different'
                              ' conductivity, all coordinates are assumed to be in meters')
                        )
    parser.add_argument("electrodefile",
                        help=('CSV with electrode names and positions, in meters, with a header of: \n'
                              '\tNAME,X,Y,Z')
                        )
    parser.add_argument("output", type=str,
                        help=("output folder or path to .vtk file with results."
                              "Results will be saved as VTK attributes with names:"
                              "potential_ELECTRODE_NAME, and correction_ELECTRODE_NAME"
                              )
                        )
    parser.add_argument('-bp', "--boundary-potential", nargs="?", type=float,
                        help="Potential value at the boundary condition in Volts",
                        default=0.0)

    parser.add_argument('-c', "--conductivities", nargs='+', type=float,
                        help=("conductivities of the physical groups (materials) in S/m,"
                              " default is [0.33, 1.65, 0.0165, 0.33, 1e-10]"
                              " which are brain tissue, CSF, skull bone, skin, air"),
                        default=[0.33, 1.65, 0.0165, 0.33, 1e-10])

    parser.add_argument('-bc', "--base-conductivity", nargs="?", type=float,
                        help="base conductivity of infinite space, for the leadfield theoretical part estimation",
                        default=0.33)

    parser.add_argument('--additional-refinement', dest='additional_refinement', action='store_true',
                        help='Enable additional uniform refinement of the mesh')
    parser.set_defaults(additional_refinement=False)

    # todo debug multiprocessing!!!!
    parser.add_argument('--multiprocessing', dest='multiprocessing', action='store_true',
                        help='Enable multiprocessing per electrode, broken rn')
    parser.set_defaults(multiprocessing=False)

    namespace = parser.parse_args()

    conductivities_vector = np.array(namespace.conductivities)
    electrodes = pd.read_csv(namespace.electrodefile)

    device = mfem.Device("cpu")
    device.Print()
    mesh = prepare_mesh(namespace.meshfile, namespace.additional_refinement)

    if len(mesh.attributes.GetDataArray()) > len(conductivities_vector):
        raise Exception("There is more materials than provided conductivities!")

    if not (mesh.attributes.GetDataArray()[:len(mesh.attributes.GetDataArray())] == (
            np.array(range(len(mesh.attributes.GetDataArray()))) + 1)).all():
        raise Exception("Mesh material indexes are not correct, they should start with 1 and increase by 1")

    if namespace.multiprocessing:
        electrode_positions = electrodes[["X", "Y", "Z"]].values
        fn = partial(mfem_solve_mesh_multiprocessing_wrap, meshfile=namespace.meshfile,
                     boundary_potential=namespace.boundary_potential,
                     conductivities=conductivities_vector,
                     refinement=namespace.additional_refinement)

        results_np = process_map(fn, electrode_positions, desc="simulating electrodes mp", chunksize=1)

        fespace = prepare_fespace(mesh)

        results = []
        for result in tqdm(results_np, desc='recovering solutions'):
            solution_gridf = mfem.GridFunction(fespace)
            solution_gridf.Assign(np.array(result))
            results.append(solution_gridf)
        # need to recreate solution in FEM
    else:
        # singlethreaded electrode sim
        results = []
        for row_id, electrode in tqdm(electrodes.iterrows(), desc="simulating electrodes"):
            electrode_position = electrode[["X", "Y", "Z"]].astype(float).values
            electrode_coeff = electrode_coefficient(electrode_position)
            result = mfem_solve_mesh(electrode_coeff, mesh, boundary_potential=namespace.boundary_potential,
                                     conductivities=conductivities_vector)
            results.append(result)

    results_correction = []
    verts = mesh.GetVertexArray()
    fespace = prepare_fespace(mesh)
    for result, electrode_position in tqdm(list(zip(results, electrodes[["X", "Y", "Z"]].astype(float).values)),
                                           desc='adding theoretical solution'):
        distance_to_electrode = np.linalg.norm(np.array(electrode_position) - verts, ord=2, axis=1)
        v_kcsd = 1.0 / (4 * np.pi * namespace.base_conductivity * distance_to_electrode)
        correction = result.GetDataArray() - v_kcsd
        correction_gridf = mfem.GridFunction(fespace)
        correction_gridf.Assign(correction)
        results_correction.append(correction_gridf)

    if namespace.output.endswith(".vtk"):
        outdir = os.path.dirname(namespace.output)
        output_filename = namespace.output
    else:
        outdir = namespace.output
        output_filename = os.path.join(outdir, os.path.splitext(os.path.basename(namespace.meshfile))[0] + '.vtk')
    os.makedirs(outdir, exist_ok=True)

    output = StringIO()

    print("Formatting output mesh")
    mesh.PrintVTK(output, 0)
    for result, electrode_name in tqdm(list(zip(results, electrodes.NAME.values)), desc='formatting output potential'):
        result.SaveVTK(output, "potential_{}".format(electrode_name), 0)

    for result, electrode_name in tqdm(list(zip(results_correction, electrodes.NAME.values)),
                                       desc='formatting output correction'):
        result.SaveVTK(output, "correction_{}".format(electrode_name), 0)

    print("Saving")
    with open(output_filename, 'w') as vtk_file:
        vtk_file.write(output.getvalue())
