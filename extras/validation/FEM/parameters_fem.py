"""
@author: mbejtka
"""
import numpy as np

sigma_roi = 0.3
sigma_slice = 0.3
sigma_saline = 1.5

# from gmsh sphere_1.geo
roivol = 114
slicevol = 111
salinevol = 108
model_base_surf = 104
model_dome_surf = 105

X, Y, Z = np.mgrid[-0.95: 0.95: 5j,
                   -0.45: 0.0: 5j,
                   -0.95: 0.95: 5j]
ele_coords = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

# dipole location - Radial
rad_dipole = {'src_pos': [0., -0.25, 0.80],
              'snk_pos': [0., -0.25, 0.85],
              'name': 'rad'}

# # dipole location - Tangential
tan_dipole = {'src_pos': [0., -0.40, 0.9],
              'snk_pos': [0., -0.35, 0.9],
              'name': 'tan'}

# # # dipole location - Mix
mix_dipole = {'src_pos': [0., -0.40, 0.80],
              'snk_pos': [0., -0.35, 0.85],
              'name': 'mix'}

dipole_list = [rad_dipole, tan_dipole, mix_dipole]
