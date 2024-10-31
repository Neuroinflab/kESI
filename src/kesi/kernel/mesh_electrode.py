import numpy as np
import pyvista
from tqdm import tqdm
from kesi.fem_utils.pyvista_resampling import pyvista_sample_points


def read_mesh_electrodes(mesh_file_path, electrode_names, electrode_positions=None, attribute_prefix='potential'):
    mesh = pyvista.read(mesh_file_path)
    electrodes = []
    if electrode_positions is None:
        for electrode in tqdm(electrode_names, desc="loading electrodes"):
            electrodes.append(MeshElectrode(mesh, electrode, attribute_prefix))
    else:
        assert len(electrode_names) == len(electrode_positions)
        for electrode, position in tqdm(zip(electrode_names, electrode_positions), desc="loading electrodes",
                                        total=len(electrode_names)):
            electrodes.append(MeshElectrode(mesh, electrode, attribute_prefix, position=position))
    return electrodes

class MeshElectrode:
    def __init__(self, pyvista_mesh, electrode_name, attribute_prefix='potential', position=(np.nan, np.nan, np.nan)):
        self.name = electrode_name
        leadfield = pyvista_mesh.point_data["{}_{}".format(attribute_prefix, electrode_name)]
        self.mesh = pyvista_mesh.copy()
        self.mesh.clear_point_data()
        self.mesh.point_data["leadfield"] = leadfield
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

    def leadfield(self, X, Y, Z):
        """Returns sampled attribute"""
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        resampled_leadfield = np.array(pyvista_sample_points(self.mesh, points).point_data['leadfield'])
        resampled_leadfield3d = resampled_leadfield.reshape(X.shape)
        return resampled_leadfield3d

    def correction_leadfield(self, X, Y, Z):
        """Returns sampled attribute, for second function compatability reasons"""
        return self.leadfield(X, Y, Z)