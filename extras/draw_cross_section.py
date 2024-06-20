import os.path

import nibabel
import numpy as np
import pandas as pd
import pylab as pb

def create_meshgrid_from_affine(affine, data):
    x, y, z, comp = data.shape

    # Create meshgrid in image voxel coordinates
    meshgrid_x, meshgrid_y, meshgrid_z = np.mgrid[0:x, 0:y, 0:z]

    # Convert meshgrid to real space coordinates
    meshgrid_coords = np.array([meshgrid_x.ravel(), meshgrid_y.ravel(), meshgrid_z.ravel(), np.ones(meshgrid_x.size)])
    real_coords = np.dot(affine, meshgrid_coords).T
    real_coords = real_coords[:, :3].reshape(meshgrid_x.shape + (3,))
    return real_coords


def main():
    electrode_file = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/bundled/electrode_locations/10_20/10_20_FOUR_SPHERES_DEPTH_L2_0.089_TEST.csv"
    sampled_potential = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/mfem_leadfield_corrections_10_20_TEST_DEPTH/sampled_potential"
    sampled_correction = "/home/mdovgialo/projects/halje_data_analysis/kESI/extras/data/generated/tutorial/four_spheres/mfem_leadfield_corrections_10_20_TEST_DEPTH/sampled_correction"
    electrodes = pd.read_csv(electrode_file)
    slice_of_interest = np.s_[130, 118, :]

    fig_corr = pb.figure()
    fig_pot = pb.figure()
    fig_pot_theor = pb.figure()
    fig_corr2 = pb.figure()

    for electrode_id in range(len(electrodes)):
        name, x, y ,z = electrodes.iloc[electrode_id]
        correction = nibabel.load(os.path.join(sampled_correction, "{}.nii.gz").format(name))
        correction_data = correction.get_fdata()
        correction_meshgrid = create_meshgrid_from_affine(correction.affine, correction_data) / 1000 # mm to meters
        correction_slice = correction_data[slice_of_interest]
        correction_x = correction_meshgrid[slice_of_interest]
        correction_x = correction_x[:, np.where((np.diff(correction_x, axis=0)!=0).all(axis=0))[0][0]].squeeze()


        potential = nibabel.load(os.path.join(sampled_potential, "{}.nii.gz").format(name))
        potential_data = potential.get_fdata()
        potential_meshgrid = create_meshgrid_from_affine(potential.affine, potential_data) / 1000 # mm to meters
        potential_slice = potential_data[slice_of_interest].squeeze()
        potential_x = potential_meshgrid[slice_of_interest]
        potential_x = potential_x[:, np.where((np.diff(potential_x, axis=0)!=0).all(axis=0))[0][0]].squeeze()

        electrode_position = np.array([x, y, z])
        distance_to_electrode = np.linalg.norm(np.array(electrode_position) - potential_meshgrid, ord=2, axis=3)
        v_kcsd = 1.0 / (4 * np.pi * 0.33 * distance_to_electrode)
        potential_theoretical = v_kcsd
        potential_theoretical_slice = potential_theoretical[slice_of_interest].squeeze()
        potential_theoretical_x = potential_meshgrid[slice_of_interest]
        potential_theoretical_x = potential_theoretical_x[:, np.where((np.diff(potential_theoretical_x, axis=0)!=0).all(axis=0))[0][0]].squeeze()

        correction2_slice = potential_slice - potential_theoretical_slice

        pb.figure(fig_corr)
        pb.plot(correction_x, correction_slice, label=name)

        pb.figure(fig_pot)
        pb.plot(potential_x, potential_slice, label=name)

        pb.figure(fig_pot_theor)
        pb.plot(potential_theoretical_x, potential_theoretical_slice, label=name)

        pb.figure(fig_corr2)
        pb.plot(potential_x, correction2_slice, label=name)

    pb.figure(fig_corr)
    pb.legend()
    pb.title("MFEM Correction")

    pb.figure(fig_pot)
    pb.legend()
    pb.title("MFEM Potential")

    pb.figure(fig_pot_theor)
    pb.legend()
    pb.title("kCSD Potential")

    pb.figure(fig_corr2)
    pb.legend()
    pb.title("Correction recalculated")

    pb.show()






if __name__ == '__main__':
    main()