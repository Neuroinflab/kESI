from memoization import cached
import numpy as np
import pyvista
from tqdm.auto import tqdm
from kesi.fem_utils.pyvista_resampling import pyvista_sample_points


def read_mesh_electrodes(mesh_file_path, electrode_names, electrode_positions=None, attribute_prefix='potential'):
    print("loading electrodes mesh...")
    mesh = pyvista.read(mesh_file_path)
    print("loading electrodes mesh done")
    electrodes = []
    if electrode_positions is None:
        for electrode in tqdm(electrode_names, desc="loading electrodes"):
            electrodes.append(MeshElectrode(mesh, electrode, attribute_prefix))
    else:
        assert len(electrode_names) == len(electrode_positions)
        for electrode, position in tqdm(zip(electrode_names, electrode_positions), desc="loading electrodes",
                                        total=len(electrode_names)):
            leadfield = mesh.point_data["{}_{}".format(attribute_prefix, electrode)]
            electrodes.append(MeshElectrode(mesh, electrode, attribute_prefix, position=position))
    return electrodes


@cached
def _resample_mesh_memoized(mesh, points):
    points = np.array(points)
    resampled_mesh = pyvista_sample_points(mesh, points)
    return resampled_mesh


class MeshElectrode:
    """
    An electrode object contains information about electrode spatial location (.x, .y and .z attribute),
     which is an absolute minimum to be used by kESI (in this case: kCSD with known
     profile of potential basis functions). It may also provide additional information about:

    medium conductivity (.conductivity attribute) normalized by the conductivity
    assumed when calculating the profile of potential basis function for kCSD, or
    leadfield (.leadfield() method) which enables
    kESI for arbitrary shape of CSD basis functions, or
    leadfield correction (.correction_leadfield() method) which enables kESI for setups
    violating kCSD assumptions, while facilitating application of analitically derived
    kCSD base functions to avoid significant numerical errors,
    base conductivity (.conductivity attribute) assumed when calculating the leadfield correction.

    MeshElectrode reads .vtk files with leadfields and resamples it to convolver._POT_ grid when called for.

   To be used with pbf.Numerical, as it defines the leadfield.

    """

    def __init__(self, pyvista_mesh, electrode_name, attribute_prefix='potential', position=(np.nan, np.nan, np.nan)):
        self.name = electrode_name
        self.mesh = pyvista_mesh
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.attribute_prefix = attribute_prefix
        # we don't add conductivity here, because it's accounted for in leadfield

    def resample(self, X, Y, Z):
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        resampled_mesh = _resample_mesh_memoized(self.mesh, points)
        return resampled_mesh

    def leadfield(self, X, Y, Z):
        """Returns sampled attribute"""
        resampled_mesh = self.resample(X, Y, Z)
        resampled_leadfield = np.array(resampled_mesh.point_data["{}_{}".format(self.attribute_prefix, self.name)])
        resampled_leadfield3d = resampled_leadfield.reshape(X.shape)
        return resampled_leadfield3d

    def correction_leadfield(self, X, Y, Z):
        """Returns sampled attribute, for second function compatability reasons"""
        return self.leadfield(X, Y, Z)
