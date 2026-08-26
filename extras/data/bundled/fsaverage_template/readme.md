# Creating the mesh

In this folder you can find partitioned MRI scans from 
FreeSurfer/MNE average head MRI *fsaverage*.

The parcellation is done in freesurfer and assembled 
semi manually, by using Blender to combine 
inner and outer skull surfaces to a one skull object,
which was then sampled in Slicer3D to boolean mask
in the  *skull.nii.gz* file to match the scan.

`combine_freesurf_parcellation.py` was used to combine 
skull, brain, head parcellations into an atlas of 5 materials:
+ brain
+ csf
+ skull
+ scalp
+ air

saved in `fsaverage_brain_materials_1mm.nii.gz` file. 
It was also resampled to 2 mm accuracy for faster calculations.

10-05 system EEG electrodes were placed into the 
scalp by taking 10-05 electrode positions from MNE, normalizing their vectors,
and then projecting these vectors from infinity 
to the surface of the skull (file `placing_10_05_electrodes.py`)

You can find those new electrode locations in
`extras/data/bundled/electrode_locations/10_20/10_20_FSAVERAGE_SCALP.csv` file.

Afterwards, the new, parcellated MRI atlas is 
transformed into MFEM compatible mesh using `kesi_mri_to_mesh` 
command. That also assigns most bottom surface as a physical 
surface and to be used as a big grounding plate.

This model afterwards can be used in forward modelling and kESI inverse CSD solving.

In case of issues with correction sampling due to electrode
placement exactly on the nodes of the mesh it is required to fix NaNs in the sampled
electrode correction using `extras/check_and_fix_corrections.py`.

To create the electrode correction using FEM model and to sample it refer to `extras/tutorial_four_spheres_mfem.ipynb`.
