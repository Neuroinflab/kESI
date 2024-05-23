import os.path

import nibabel
import numpy as np
import pylab as pb

files = [
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first+potential.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finer/first+potential.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_fine/first+potential.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_normal/first+potential.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first_potential_only.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections/first_potential_only.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.005.msh_potential.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_with_electrode_0.002.msh_potential.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.005.msh_potential_theory.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first_potential_only.nii.gz",
]

# files = [
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finer/first.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_fine/first.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_normal/first.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.005.msh_potential_combined.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_with_electrode_0.002.msh_potential_combined.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_with_electrode_0.005.msh_potential_combined.nii.gz",
# ]

files = [
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_with_electrode_0.002.msh_potential_combined.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential_combined.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first.nii.gz",
]

files = [
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first_potential_only.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first+potential.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential_theory.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_with_electrode_0.002.msh_potential_combined.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential_combined.nii.gz",
]


files = [
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential.nii.gz",
    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_with_electrode_0.002.msh_potential_combined.nii.gz",

    # "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential_theory.nii.gz",
"/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first+potential.nii.gz",
    "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/first_potential_only.nii.gz"
]

files = [

"/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.005.msh_potential_theory.nii.gz",
"/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential_theory.nii.gz",
"/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.005.msh_potential_theory_sphere.nii.gz",
"/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/four_spheres_in_air_with_plane_0.002.msh_potential_theory_sphere.nii.gz",
"/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/sampled_leadfield_corrections_finest/theoretic_potenial_dx_0.00078.nii.gz",

]
for file in files:

    a = nibabel.load(file)
    print (a.affine)

    z = int(a.shape[2] / 2)
    y = int(a.shape[1] / 2)
    x = int(a.shape[0] / 2)

    data = a.get_fdata()
    vertical_line = data[x, y, :]
    z_line_coords = np.arange(a.affine[2][3], a.affine[2][3] + a.shape[2] * a.affine[2][2], a.affine[2][2])
    baseline = np.median(vertical_line[(z_line_coords<0.02) * (z_line_coords>-0.02)])

    pb.plot(z_line_coords, vertical_line-baseline, label=os.path.basename(file))

pb.axvline(0.0785, color='k', label='electrode', alpha=0.2)
pb.axvline(0.079, linestyle='--', label='brain', alpha=0.2)
pb.axvline(0.082, linestyle='--', label='CSF', alpha=0.2)
pb.axvline(0.086, linestyle='--', label='skull', alpha=0.2)
pb.axvline(0.09, linestyle='--', label='skin', alpha=0.2)


pb.legend()
pb.show()