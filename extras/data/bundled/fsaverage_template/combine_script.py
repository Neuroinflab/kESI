import os.path

import nibabel
import numpy as np
import pandas as pd
from skimage.segmentation import flood_fill

current_dir = os.path.dirname(__file__)

freesurf_lut = pd.read_csv(os.path.join(current_dir, "freesurfercolorLUT_cleaned_itksnap.label"), delimiter='\t',
                           header=None, names=['id', 'r', 'g', 'b', 'a', 'vis', 'vis3d', 'label'])

materials_lut = pd.read_csv(os.path.join(current_dir, "materials.label"), delimiter='\t',
                           header=None, names=['id', 'r', 'g', 'b', 'a', 'vis', 'vis3d', 'label'])
materials_lut_dict = {}
for id, row in materials_lut.iterrows():
    materials_lut_dict[row.label] = row.id

csf_list = [
    'Right-Lateral-Ventricle',
    'Left-Lateral-Ventricle',
    "3rd-Ventricle",
    '4th-Ventricle',
]

freesurf_to_mat = {}
for id, row in freesurf_lut.iterrows():
    material = 'CSF'

    if row.id > 0:
        material = "brain"
    print(row.label)
    print(row.label in csf_list)
    if row.label in csf_list:
        material = 'CSF'
    # print(row, material)
    freesurf_to_mat[row.id] = materials_lut_dict[material]

parcelation_brain = nibabel.load(os.path.join(current_dir, "aseg.mgz.nii.gz"))
parcelation_skull = nibabel.load(os.path.join(current_dir, "skull.nii.gz"))
parcelation_head = nibabel.load(os.path.join(current_dir, "seghead.mgz.nii.gz"))
t1_ref = nibabel.load(os.path.join(current_dir, "T1.mgz.nii.gz"))



final_parcellation = np.zeros_like(t1_ref.get_fdata()).astype(np.uint8)
final_parcellation[:] = materials_lut_dict['air']
final_parcellation[parcelation_head.get_fdata()>0] = materials_lut_dict['scalp']
final_parcellation[parcelation_skull.get_fdata()>0] = materials_lut_dict['skull']

final_parcellation = flood_fill(final_parcellation, (128, 128, 128), materials_lut_dict['CSF'])

inside_skull = (final_parcellation == materials_lut_dict['CSF'])

parcelation_brain_data = parcelation_brain.get_fdata().astype(np.uint32)
brain_material = np.zeros_like(parcelation_brain_data, dtype=parcelation_brain_data.dtype)

for parcelation in np.unique((parcelation_brain_data)):
    material = freesurf_to_mat[parcelation]
    brain_material[parcelation_brain_data==parcelation] = material
final_parcellation[inside_skull] = brain_material[inside_skull]


final_image = nibabel.Nifti1Image(final_parcellation, header=t1_ref.header, affine=t1_ref.affine)
nibabel.save(final_image, os.path.join(current_dir, "materials.nii.gz"))




import IPython
IPython.embed()