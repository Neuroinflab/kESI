import os.path

import nibabel
import numpy as np
import pandas as pd
from nibabel import Nifti1Image

from kesi.common import GaussianSourceKCSD3D
from kesi.kesi import Kesi3d, KcsdKesi3d
from kesi.mfem_solver.forward_solver import CSDForwardSolver
import logging

from ncsd_tools.grid_utils import create_atlas_addon_nii_with_circles


def normalize_eigensources(kcsd_eigensources):
    for i in range(kcsd_eigensources.shape[-1]):
        kcsd_eigensources[:, :, :, i] /= np.max(np.abs(kcsd_eigensources[:, :, :, i]))
    return kcsd_eigensources

logger = logging.getLogger("test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
this_folder = os.path.dirname(__file__)


# meshfile = os.path.join(this_folder, "data", "generated", "meshes", "four_spheres_with_electrode_0.005.msh")
meshfile = os.path.join(this_folder, "data", "generated", "meshes", "four_spheres_in_air_with_plane_0.005.msh")
conductivities = np.array([0.33, 1.65, 0.0165, 0.33, 1e-10])
# conductivities = np.array([0.33, 1.65, 0.0165, 0.33, 1e-99])

# electrodes_file = os.path.join(this_folder, "data", "generated", "tutorial", "four_spheres", "tutorial_electrodes_four.csv")

electrodes_file = os.path.join(this_folder, "data", "bundled", "electrode_locations", "10_20", "10_20_FOUR_SPHERES_DEPTH_L2_0.089.csv")
# electrodes_file = os.path.join(this_folder, "data", "bundled", "electrode_locations", "10_20", "10_20_MINIMUM_FOUR_SPHERES_SCALP_0.089.csv")
electrodes_df = pd.read_csv(electrodes_file)

electrode_correction_sampling_folder = os.path.join("data",
                                                    "generated",
                                                    "tutorial",
                                                    "four_spheres",
                                                    "mfem_sampled_leadfield_corrections_air_10_20_big_L2_v2"
                                                    )

sampling_points = electrodes_df[["X", "Y", "Z"]].values

electrode_names = electrodes_df["NAME"].to_list()

solver = CSDForwardSolver(meshfile, conductivities)

mesh_points = np.array(solver.mesh.GetVertexArray())

min_x = np.min(mesh_points[:, 0])
max_x = np.max(mesh_points[:, 0])
min_y = np.min(mesh_points[:, 1])
max_y = np.max(mesh_points[:, 1])
min_z = np.min(mesh_points[:, 2])
max_z = np.max(mesh_points[:, 2])

dx = 0.001

x = np.arange(min_x, max_x, dx)
y = np.arange(min_y, max_y, dx)
z = np.arange(min_z, max_z, dx)

grid = [x, y, z]

meshgrid = np.meshgrid(x, y, z, indexing='ij')
ground_truth = np.zeros((len(x), len(y), len(z)))

test_dipole = np.array([0, 0, 0.072])
test_dipole_dir = np.array([1, 0, 0])
seperation = 0.001
size = 0.002

posp = test_dipole + test_dipole_dir * seperation
posn = test_dipole - test_dipole_dir * seperation

pos_s = GaussianSourceKCSD3D(posp[0], posp[1], posp[2], size, conductivity=0.33)
neg_s = GaussianSourceKCSD3D(posn[0], posn[1], posn[2], size, conductivity=0.33)

ground_truth += pos_s.csd(*meshgrid)
ground_truth -= neg_s.csd(*meshgrid)

solution = solver.solve(grid, ground_truth)

logger.info("Sampling points")
potential = solver.sample_solution(sampling_points)
print(potential)
logger.info("Sampling points Done")

meshgird_collate = np.rollaxis(np.array(meshgrid), 0, 4)
logger.info("Sampling potential grid")
potential_grid = solver.sample_solution(meshgird_collate)
logger.info("Sampling potential grid done")
brain_mask = ((np.sum(meshgird_collate ** 2, axis=3)**0.5) < 0.079)

logger.info("Starting kesi solver")

solution_dx = 0.005
sol_x = np.arange(min_x, max_x, solution_dx)
sol_y = np.arange(min_y, max_y, solution_dx)
sol_z = np.arange(min_z, max_z, solution_dx)
solution_meshrid = np.meshgrid(sol_x, sol_y, sol_z, indexing='ij')


solution_affine = np.zeros((4, 4))
solution_affine[0][0] = np.abs(solution_dx)
solution_affine[1][1] = np.abs(solution_dx)
solution_affine[2][2] = np.abs(solution_dx)
# last row is always 0 0 0 1

new_000 = np.array([solution_meshrid[0][0, 0 ,0], solution_meshrid[1][0, 0 ,0], solution_meshrid[2][0, 0 ,0]])
solution_affine[0][3] = new_000[0]
solution_affine[1][3] = new_000[1]
solution_affine[2][3] = new_000[2]

solution_affine = solution_affine * 1000  # to mm
solution_affine[3][3] = 1

meshgird_collate_sol = np.rollaxis(np.array(solution_meshrid), 0, 4)
brain_mask_sol = ((np.sum(meshgird_collate_sol ** 2, axis=3)**0.5) < 0.079)


kesi_solver = Kesi3d(solution_meshrid, electrode_names, electrode_correction_sampling_folder,
                    conductivity=0.33, R_init=0.012, mask=brain_mask_sol)
#
# import IPython
# IPython.embed()


logger.info("Starting kesi solver cv lambda")
#best_lambda, lambd, errors = kesi_solver.cv_lambda(potential[:, None])
best_lambda, lambd, errors = kesi_solver.cv_lambda(potential[:, None], np.logspace(-7, 20, 100))

logger.info("Starting kesi reconstructing")
reconstructed_csd = kesi_solver.reconstruct_csd(potential[:, None], best_lambda)

logger.info("Starting kesi eigenvectors")
kesi_eigenvalues, kesi_eigensources = kesi_solver.eigh(best_lambda)

kesi_eigensources = normalize_eigensources(kesi_eigensources)


csd_grid = reconstructed_csd * 1e9 / 1e3 ** 3  # transform from Amper per meter cubed to nanoAmpers per milimiter cubed

sdx = sdy = sdz = dx

new_affine = np.zeros((4, 4))
new_affine[0][0] = np.abs(sdx)
new_affine[1][1] = np.abs(sdy)
new_affine[2][2] = np.abs(sdz)
# last row is always 0 0 0 1

new_000 = np.array([meshgrid[0][0, 0 ,0], meshgrid[1][0, 0 ,0], meshgrid[2][0, 0 ,0]])
new_affine[0][3] = new_000[0]
new_affine[1][3] = new_000[1]
new_affine[2][3] = new_000[2]

new_affine = new_affine * 1000  # to mm
new_affine[3][3] = 1


img = Nifti1Image(csd_grid, solution_affine)

img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_kesi"))

size_on_disk = csd_grid.size * csd_grid.itemsize / 1024 / 1024 / 1024
print("Saving output Nifti, shape: {} dtype: {} maximum size on disk {:0.3f} Gb".format(csd_grid.shape,
                                                                                        csd_grid.dtype,
                                                                                        size_on_disk
                                                                                        ))
nibabel.save(img, base_path + '.nii.gz')

create_atlas_addon_nii_with_circles(sampling_points * 1000, size=7, atlas_type=base_path + '.nii.gz',
                                    savedir=os.path.dirname(base_path), title="electrodes"
                                    )


img = Nifti1Image(kesi_eigensources, solution_affine)

img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_kesi_eigensources"))

size_on_disk = csd_grid.size * csd_grid.itemsize / 1024 / 1024 / 1024
print("Saving output Nifti, shape: {} dtype: {} maximum size on disk {:0.3f} Gb".format(csd_grid.shape,
                                                                                        csd_grid.dtype,
                                                                                        size_on_disk
                                                                                        ))
nibabel.save(img, base_path + '.nii.gz')



img = Nifti1Image(ground_truth[:, :, :, None], new_affine)
img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_gt"))

nibabel.save(img, base_path + '.nii.gz')


img = Nifti1Image(potential_grid[:, :, :, None], new_affine)
img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_potential"))

nibabel.save(img, base_path + '.nii.gz')



kcsd_solver = KcsdKesi3d(solution_meshrid, sampling_points,
                    conductivity=0.33, R_init=0.012, mask=brain_mask_sol)


best_lambda, lambd, errors = kcsd_solver.cv_lambda(potential[:, None], np.logspace(-7, 20, 100))
reconstructed_csd = kcsd_solver.reconstruct_csd(potential[:, None], best_lambda)

kcsd_eigenvalues, kcsd_eigensources = kcsd_solver.eigh(best_lambda)

kcsd_eigensources = normalize_eigensources(kcsd_eigensources)

csd_grid = reconstructed_csd * 1e9 / 1e3 ** 3  # transform from Amper per meter cubed to nanoAmpers per milimiter cubed

img = Nifti1Image(csd_grid, solution_affine)
img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_kcsd"))

size_on_disk = csd_grid.size * csd_grid.itemsize / 1024 / 1024 / 1024
print("Saving output Nifti, shape: {} dtype: {} maximum size on disk {:0.3f} Gb".format(csd_grid.shape,
                                                                                        csd_grid.dtype,
                                                                                        size_on_disk
                                                                                        ))
base_path = os.path.expanduser(os.path.join("~/test", "test_kcsd"))
nibabel.save(img, base_path + '.nii.gz')

img = Nifti1Image(kcsd_eigensources.astype(np.float32), solution_affine)

img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_kcsd_eigensources"))

size_on_disk = kcsd_eigensources.size * csd_grid.itemsize / 1024 / 1024 / 1024
print("Saving output Nifti, shape: {} dtype: {} maximum size on disk {:0.3f} Gb".format(kcsd_eigensources.shape,
                                                                                        csd_grid.dtype,
                                                                                        size_on_disk
                                                                                        ))
nibabel.save(img, base_path + '.nii.gz')


img = Nifti1Image(brain_mask[:,:,:,None].astype(np.float32), new_affine)
img.header.set_xyzt_units(xyz=2, t=24)  # mm
zooms = list(img.header.get_zooms())
sample_period_us = (1 / 1000) * 1e6
zooms[-1] = sample_period_us
img.header.set_zooms(tuple(zooms))

base_path = os.path.expanduser(os.path.join("~/test", "test_brain_mask"))

size_on_disk = csd_grid.size * csd_grid.itemsize / 1024 / 1024 / 1024
print("Saving output Nifti, shape: {} dtype: {} maximum size on disk {:0.3f} Gb".format(csd_grid.shape,
                                                                                        csd_grid.dtype,
                                                                                        size_on_disk
                                                                                        ))

nibabel.save(img, base_path + '.nii.gz')




import IPython
IPython.embed()