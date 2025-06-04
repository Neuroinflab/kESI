import argparse
import math
import os
from functools import partial, lru_cache
from multiprocessing import set_start_method

import mfem.ser as mfem
import numpy as np
import pandas as pd
import psutil
import pyvista
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from io import StringIO


from kesi.fem_utils.vtk_utils import grid_function_save_vtk
from kesi.mfem_solver.interpolated_mfem_coefficient import CSDCoefficient
from kesi.utils import str_to_bool, write_run_summary
from scipy.spatial import KDTree


def calculate_refinement_error(mesh, electrode_positions, refinement_radius):
    vertices = np.array(mesh.GetVertexArray())
    centroids = []
    for i in tqdm(range(mesh.GetNE()), total=mesh.GetNE(), desc='calculating element centroids'):
        element = mesh.GetElement(i)
        centroid = vertices[element.GetVerticesArray()].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    tree = KDTree(centroids)
    error = np.zeros(mesh.GetNE())

    for electrode_position in tqdm(electrode_positions, desc='setting elements to refine'):
        nearby = tree.query_ball_point(electrode_position, refinement_radius)
        error[nearby] = 1.0

    error_mfem =  mfem.Vector(mesh.GetNE())
    error_mfem.Assign(error)
    return error_mfem


def refine_around_electrodes(mesh, electrode_positions, refinement_radius=0.01, steps=2):
    """refines the mesh around electrode_positions (tuple of tuples[len 3]) with mesh element centroids in refinement_radius,
    performs that steps times."""
    for i in range(steps):
        # everything close enough to electrode positions is marked as 1, far away as 0
        error_mfem = calculate_refinement_error(mesh, electrode_positions, refinement_radius)
        # we refine all elements who's error is higher than 0.5
        mesh.RefineByError(error_mfem, 0.5, 0, 0)
        mesh.Finalize()
    return mesh


@lru_cache
def prepare_mesh(meshfile, refinement, electrode_positions=None, refinement_radius=0.01, steps=2):
    """
    Params:

    :param meshfile: path to MFEM compatible mesh file
    :param refinement: - bool - set to true to uniformly refine mesh
    :param electrode_positions: - set to None to do nothing, set to tuple of tuples, containing electrode positions to refine the mesh around the electrodes
        Example: ((0, 0,0), (1,1,1))
    :param refinement_radius: - float, how much space around the electrodes to refine, all element which centroids fit into the radius will get refined.
        If the mesh is too coarse there might be no mesh element in the refinement radius, you might want to increase the radius
    :param steps: how many times we want to do the refinement around electrode positions.
    """
    # to create run
    # gmsh -3 -format msh22 four_spheres_in_air_with_plane.geo
    print("Loading mesh...")
    mesh = mfem.Mesh(meshfile, 1, 1)
    print("Loading mesh... Done")

    if refinement:
        print("additional uniform refinement...")
        mesh.UniformRefinement()
        print("additional uniform refinement... Done")

    if electrode_positions is not None:
        mesh = refine_around_electrodes(mesh, electrode_positions, refinement_radius=refinement_radius, steps=steps)

    return mesh


def prepare_fespace(mesh):
    dim = mesh.Dimension()
    order = 1
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec)
    return fespace


def mfem_solve_mesh_multiprocessing_wrap(electrode_position, boundary_potential, conductivities,
                                         meshfile,
                                         electrodes_for_prepare,
                                         refinement,
                                         refinement_radius,
                                         refinement_steps,
                                         ):
    try:
        device = mfem.Device("cpu")
        device.Print()
    except RuntimeError:
        pass  # already configured
    coeff = electrode_coefficient(electrode_position)
    mesh = prepare_mesh(meshfile, refinement, electrodes_for_prepare, refinement_radius, refinement_steps)
    result = mfem_solve_mesh(coeff, mesh, boundary_potential, conductivities)
    sol = np.array(result.GetDataArray())
    return sol


def electrode_coefficient(electrode_position, scale=1):
    # for each point charge
    point_charge_coeff = mfem.DeltaCoefficient()
    point_charge_coeff.SetScale(scale)
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


def mfem_solve_mesh(csd_coefficient, mesh, boundary_potential, conductivities, dirichlet=True):
    """
    csd_coefficient - CSD distribution in coefficient form
    mesh - MFEM mesh object
    boundary_potential - value of the potential at the ground
    conductivities - numpy array of conductivities in S/m one per mesh material, can be longer than amount of materials - extra values won't not be used
    dirichlet - boolean - to enable dirichlet boundary condition, otherwise it's neuman - and boundary_potential is the current through boundary
    """

    # import IPython
    # IPython.embed()

    # this fespace will get garbage collected and returned gridfunctions will crash on some operations!!!!!
    fespace = prepare_fespace(mesh)
    print('Number of finite element unknowns: ' +
          str(fespace.GetTrueVSize()))

    conductivities_vector = mfem.Vector(list(conductivities))
    conductivities_coeff = mfem.PWConstCoefficient(conductivities_vector)

    # this is a masking list which decides, which boundaries of the mesh are Dirichlet boundaries
    # in this case we have 4 spheres, and for some reason index of the outside boundary is 5
    # for now I want to set the outside boundary of the 4 spheres as essential boundary and having a 0 potential.
    # dirichlet boundary
    ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
    ess_bdr.Assign(0)
    # if dirichlet:
    ess_bdr[mesh.bdr_attributes[-1] - 1] = 1
    # if Neuman should be an empty list
    ess_tdof_list = mfem.intArray()
    if dirichlet:
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    b = mfem.LinearForm(fespace)
    if isinstance(csd_coefficient, list):
        for i in csd_coefficient:
            b.AddDomainIntegrator(mfem.DomainLFIntegrator(i))
    else:
        b.AddDomainIntegrator(mfem.DomainLFIntegrator(csd_coefficient))

    if not dirichlet:
        # Define Neumann boundary function
        g = mfem.ConstantCoefficient(boundary_potential)
        b.AddBoundaryIntegrator(mfem.BoundaryLFIntegrator(g), ess_bdr)
    b.Assemble()

    x = mfem.GridFunction(fespace)
    # setting initial values in all points, boundary elements will enforce this  value
    # if they are set to be eesential True Dofs (dirichlet conditions) when Forming a linear system
    x.Assign(float(boundary_potential))

    a = mfem.BilinearForm(fespace)

    a.AddDomainIntegrator(mfem.DiffusionIntegrator(conductivities_coeff))
    a.Assemble()

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()

    # ess_tdof_list is a list of elements which are supposed to be already solved, Dirichlet boundary condition
    # if it's an empty list then there is no enforced potential value. Point sources will not converge
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    AA = A.AsSparseMatrix()
    M = mfem.GSSmoother(AA)
    mfem.PCG(A, M, B, X, 1, 10000, 1e-12, 0.0)

    a.RecoverFEMSolution(X, b, x)

    return x

def estimate_sensible_process_count(safety_margin=3):
    """
    Call this function just before starting clones of your process.
    Assumes that clones will use similar amount of memory time safety margin.
    Makes sure to return amount of processes which will fit in free RAM
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    resident_memory_bytes = mem_info.rss
    resident_memory_needed_bytes = resident_memory_bytes * safety_margin
    mem = psutil.virtual_memory()
    available_ram_bytes = mem.available
    proces_num = int(math.floor(available_ram_bytes / resident_memory_needed_bytes))

    print(("PROCESS COUNT ESTIMATION: Avialable RAM: {:.2f} Gb, this"
          " process takes {:.2f} Gb of RAM, including safety margin of {} - we will spawn {} "
           "processes").format(available_ram_bytes / 1024 ** 3, resident_memory_needed_bytes / 1024 **3,
                               safety_margin, proces_num)
          )
    if proces_num > os.cpu_count():
        proces_num = os.cpu_count()
        print("however this computer has only {} cores, reducing process number to {}".format(os.cpu_count(),
                                                                                              os.cpu_count())
              )
    return proces_num


def main():
    parser = argparse.ArgumentParser(description="samples mesh solution using voxel downsampling")
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

    parser.add_argument("--save-correction", type=str_to_bool,
                        help="y/n save solved correction per electrode",
                        default=False)

    parser.add_argument("--save-potential", type=str_to_bool,
                        help="y/n save solved potential per electrode",
                        default=True)

    parser.add_argument('--additional-refinement', dest='additional_refinement', action='store_true',
                        help='Enable additional uniform refinement of the mesh')
    parser.set_defaults(additional_refinement=False)

    parser.add_argument("--electrode-refinement", type=float,
                        help=("Refine mesh around electrode points, radius in meters. Leave unset for no refinement. "
                              "If the mesh is too coarse there might be no element centroids in the radius. "
                              "Set the radius appropriately. Defaults to no refinement."),
                        default=None)

    parser.add_argument("--electrode-refinement-steps", type=int,
                        help=("When doing mesh refinement around electrodes, define amount of refinement steps. Each step subdivides the elements near electrodes."
                              " Defaults to 2."),
                        default=2)

    parser.add_argument('--multiprocessing', dest='multiprocessing', action='store_true',
                        help='Enable multiprocessing per electrode')
    parser.set_defaults(multiprocessing=False)

    parser.add_argument('-pc', "--process-count", type=int,
                        help=("if multiprocessing is enabled sets the amount of cores to use,"
                              " default as much CPU cores as possible to fit into RAM"),
                        default=None)

    parser.add_argument("--save-vtk", type=str_to_bool,
                        help="y/n save solved correction per electrode in the vtk file",
                        default=True)

    parser.add_argument("--save-numpy", type=str_to_bool,
                        help=("y/n save mesh solution in a series of numpy files,"
                              " saves a lot of space compared to VTK, but cannot"
                              " be previewed using industry standard mesh viewers, can still be used for sampling"),
                        default=False)

    parser.add_argument("--numpy-precision", type=np.dtype,
                        help=("Dtype precision for numpy solution saving, in numpy dtype understood string form. ie: "
                              "float32 float64 etc"),
                        default=np.dtype("float32"))

    namespace = parser.parse_args()
    if namespace.multiprocessing:
        set_start_method("spawn")

    if not (namespace.save_potential or namespace.save_correction):
        raise Exception("Nothing will be saved! Exiting")

    if not (namespace.save_vtk or namespace.save_numpy):
        raise Exception("Nothing will be saved! Exiting")

    write_run_summary(savedir=namespace.output, namespace=namespace)

    conductivities_vector = np.array(namespace.conductivities)
    electrodes = pd.read_csv(namespace.electrodefile)

    device = mfem.Device("cpu")
    device.Print()
    if namespace.electrode_refinement:
        electrodes_for_prepare = electrodes[["x", "y", "z"]].values / 1000  # electrodes in mm, mesh in meters
        electrodes_for_prepare = tuple((tuple(i) for i in electrodes_for_prepare))
    else:
        electrodes_for_prepare = None
    mesh = prepare_mesh(namespace.meshfile, namespace.additional_refinement, electrodes_for_prepare, refinement_radius=namespace.electrode_refinement,
                        steps=namespace.electrode_refinement_steps)
    # fespace might need to exist all the time for GridFunctions to work, if fespace gets eaten by garbage collector
    # GF crashes
    fespace = prepare_fespace(mesh)

    outdir = namespace.output
    output_filename = os.path.join(outdir, os.path.splitext(os.path.basename(namespace.meshfile))[0] + '.vtk')
    os.makedirs(outdir, exist_ok=True)

    # todo: at 0.002 max element size for spheres with plane it only saves 16 megabytes of mesh, I don't understand why
    # it happens in any mode of PrintVTK, even directly to file
    # MAYBE FIXED BY REIMPLEMENTING SAVING OF THE GRID FUNCTIONS!!!
    output = StringIO()
    print("saving output mesh")
    mesh.PrintVTK(output)
    with open(output_filename, 'w') as vtk_file:
        vtk_file.write(output.getvalue())
    del output

    if len(mesh.attributes.GetDataArray()) > len(conductivities_vector):
        raise Exception("There is more materials than provided conductivities!")

    if not (mesh.attributes.GetDataArray()[:len(mesh.attributes.GetDataArray())] == (
            np.array(range(len(mesh.attributes.GetDataArray()))) + 1)).all():
        raise Exception("Mesh material indexes are not correct, they should start with 1 and increase by 1")

    if namespace.multiprocessing:
        electrode_positions = electrodes[["x", "y", "z"]].values / 1000  # electrodes in mm, mesh in meters
        fn = partial(mfem_solve_mesh_multiprocessing_wrap,
                     boundary_potential=namespace.boundary_potential,
                     conductivities=conductivities_vector,
                     meshfile=namespace.meshfile,
                     electrodes_for_prepare=electrodes_for_prepare,
                     refinement=namespace.additional_refinement,
                     refinement_radius=namespace.electrode_refinement,
                     refinement_steps=namespace.electrode_refinement_steps
                     )

        if namespace.process_count is not None:
            assert isinstance(namespace.process_count, int)
            assert namespace.process_count > 0
            process_count = namespace.process_count
        else:
            # need a safety margin of 3 - in empirical testing setting up the FEM solution space, takes around
            # 3 times more ram than the loaded fem space and the mesh.
            process_count = estimate_sensible_process_count(safety_margin=3)
            assert process_count > 0
        results_np = process_map(fn, electrode_positions, desc="simulating electrodes mp", chunksize=1,
                                 max_workers=process_count)

    else:
        # singlethreaded electrode sim
        results_np = []
        for row_id, electrode in tqdm(electrodes.iterrows(), desc="simulating electrodes", total=len(electrodes)):
            # electrodes in mm, mesh in meters
            electrode_position = electrode[["x", "y", "z"]].astype(float).values / 1000
            electrode_coeff = electrode_coefficient(electrode_position)
            result = mfem_solve_mesh(electrode_coeff, mesh, boundary_potential=namespace.boundary_potential,
                                     conductivities=conductivities_vector)
            results_np.append(np.array(result.GetDataArray()))

    # due to WEIRD pyMFEM behaviour grid functions can loose their associated fespace, or mesh or whatnot and just SEGFAULT
    # to fight it need to recreate fespace or use the one which will be availabe all the time and recreate the gridfunction
    # todo report it???? Find minimal example?
    results = []
    for result in tqdm(results_np, desc='recovering solutions'):
        solution_gridf = mfem.GridFunction(fespace)
        solution_gridf.Assign([float(i) for i in list(result.copy())])
        results.append(solution_gridf)

    results_correction = []
    verts = mesh.GetVertexArray()

    for result, electrode_position in tqdm(list(zip(results, electrodes[["x", "y", "z"]].astype(float).values / 1000)),
                                           desc='adding theoretical solution'):
        distance_to_electrode = np.linalg.norm(np.array(electrode_position) - verts, ord=2, axis=1)
        v_kcsd = 1.0 / (4 * np.pi * namespace.base_conductivity * distance_to_electrode)
        correction = result.GetDataArray() - v_kcsd
        correction_gridf = mfem.GridFunction(fespace)
        correction_gridf.Assign(correction)
        results_correction.append(correction_gridf)

    if namespace.save_vtk:
        with open(output_filename, 'a') as vtk_file:
            vtk_file.write("POINT_DATA " + str(mesh.GetNV()) + "\n")

    if namespace.save_potential:
        for result, electrode_name in tqdm(list(zip(results, electrodes['label'].values)),
                                           desc='saving output potential'):
            name = "potential_{}".format(electrode_name)
            if namespace.save_vtk:
                with open(output_filename, 'a') as vtk_file:
                    grid_function_save_vtk(result, vtk_file, name)

            if namespace.save_numpy:
                data_vtk = np.array(result.GetDataArray())
                numpy_name = os.path.join(os.path.dirname(output_filename), name)
                np.savez_compressed(numpy_name, sol=data_vtk.astype(namespace.numpy_precision))

    if namespace.save_correction:
        for result, electrode_name in tqdm(list(zip(results_correction, electrodes['label'].values)),
                                           desc='saving output correction'):
            name = "correction_{}".format(electrode_name)

            if namespace.save_vtk:
                with open(output_filename, 'a') as vtk_file:
                    grid_function_save_vtk(result, vtk_file, name)

            if namespace.save_numpy:
                data_vtk = np.array(result.GetDataArray())
                numpy_name = os.path.join(os.path.dirname(output_filename), name)
                np.savez_compressed(numpy_name, sol=data_vtk.astype(namespace.numpy_precision))

    # use pyvista to rewrite VTK in binary form
    if namespace.save_vtk:
        print("Resaving in binary")
        pyvista_mesh = pyvista.read(output_filename)
        pyvista_mesh.save(output_filename)
