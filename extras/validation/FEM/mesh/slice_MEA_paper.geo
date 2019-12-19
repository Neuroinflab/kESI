//updated code of slice and saline model from Ness, T.V., et al. (Neuroinformatics 2015)

slice_thickness = 0.3;
saline_beneath = 0.02;
cortex_width = 1.0;
cortex_length = 3.0;
slice_bottom = saline_beneath - (slice_thickness/2);

cylinder_radius = 8.;
cylinder_height = 8.;

meshSaline = 1.5;
meshCortex = .03; 
meshCortexLower = 0.006;

Include 'slice_saline.geo';
//Include 'slice.geo';
Include 'cylinder.geo';
lCylinderLineLoop = newll; Line Loop(lCylinderLineLoop) = {lCylinder3, lCylinder4, lCylinder1, lCylinder2};

lSliceLineLoop = newll; Line Loop(lSliceLineLoop) = {l4b1b, l1b5b, l5b8b, l8b4b}; // electrode plane
//lSliceLineLoop = newll; Line Loop(lSliceLineLoop) = {l41, l15, l58, l84}; // electrode plane

lCylinderSurf = news; Surface(lCylinderSurf) = {lCylinderLineLoop, lSliceLineLoop};
salineSurfaceLoop = newsl; Surface Loop(salineSurfaceLoop) = {cylinderSurf1,cylinderSurf2,cylinderSurf3,
cylinderSurf4, uCylinderSurf, lCylinderSurf, sS1, sS2, sS3, sS4, sS5};
salineVolume = newv; Volume(salineVolume) = {salineSurfaceLoop};
Physical Volume(1) = {salineVolume};
Physical Volume(2) = {vSlice}; // slice volume
Physical Surface(9) = {cylinderSurf1, cylinderSurf2, cylinderSurf3, cylinderSurf4, uCylinderSurf};