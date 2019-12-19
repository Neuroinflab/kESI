

//CYLINDER
pCylinder0 = newp; Point(pCylinder0) = {-slice_thickness/2, 0, 0, meshSaline};
pCylinder1 = newp; Point(pCylinder1) = {-slice_thickness/2, 0, cylinder_radius, meshSaline};
pCylinder2 = newp; Point(pCylinder2) = {-slice_thickness/2, cylinder_radius, 0, meshSaline};
pCylinder3 = newp; Point(pCylinder3) = {-slice_thickness/2, 0, -cylinder_radius, meshSaline};
pCylinder4 = newp; Point(pCylinder4) = {-slice_thickness/2, -cylinder_radius, 0, meshSaline};
pCylinder5 = newp; Point(pCylinder5) = {cylinder_height, 0, cylinder_radius, meshSaline};
pCylinder6 = newp; Point(pCylinder6) = {cylinder_height, 0, 0, meshSaline};
pCylinder7 = newp; Point(pCylinder7) = {cylinder_height, cylinder_radius, 0, meshSaline};
pCylinder8 = newp; Point(pCylinder8) = {cylinder_height, 0, -cylinder_radius, meshSaline};
pCylinder9 = newp; Point(pCylinder9) = {cylinder_height, -cylinder_radius, 0, meshSaline};

lCylinder1 = newl; Circle(lCylinder1) = {pCylinder1, pCylinder0, pCylinder2};
lCylinder2 = newl; Circle(lCylinder2) = {pCylinder2, pCylinder0, pCylinder3};
lCylinder3 = newl; Circle(lCylinder3) = {pCylinder3, pCylinder0, pCylinder4};
lCylinder4 = newl; Circle(lCylinder4) = {pCylinder4, pCylinder0, pCylinder1};
lCylinder5 = newl; Circle(lCylinder5) = {pCylinder5, pCylinder6, pCylinder7};
lCylinder6 = newl; Circle(lCylinder6) = {pCylinder7, pCylinder6, pCylinder8};
lCylinder7 = newl; Circle(lCylinder7) = {pCylinder8, pCylinder6, pCylinder9};
lCylinder8 = newl; Circle(lCylinder8) = {pCylinder9, pCylinder6, pCylinder5};
lCylinder9 = newl; Line(lCylinder9) = {pCylinder1, pCylinder5};
lCylinder10 = newl; Line(lCylinder10) = {pCylinder2, pCylinder7};
lCylinder11 = newl; Line(lCylinder11) = {pCylinder3, pCylinder8};
lCylinder12 = newl; Line(lCylinder12) = {pCylinder4, pCylinder9}; 

cylinderLineLoop1  = newll; Line Loop(cylinderLineLoop1) = {lCylinder7, -lCylinder12, -lCylinder3, lCylinder11};
cylinderSurf1 = news; Surface(cylinderSurf1) = {cylinderLineLoop1};
cylinderLineLoop2  = newll; Line Loop(cylinderLineLoop2) = {lCylinder11, -lCylinder6, -lCylinder10, lCylinder2};
cylinderSurf2 = news; Surface(cylinderSurf2) = {cylinderLineLoop2};
cylinderLineLoop3  = newll; Line Loop(cylinderLineLoop3) = {lCylinder10, -lCylinder5, -lCylinder9, lCylinder1};
cylinderSurf3 = news; Surface(cylinderSurf3) = {cylinderLineLoop3};
cylinderLineLoop4  = newll; Line Loop(cylinderLineLoop4) = {lCylinder9, -lCylinder8, -lCylinder12, lCylinder4};
cylinderSurf4 = news; Surface(cylinderSurf4) = {cylinderLineLoop4};
uCylinderLineLoop = newll; Line Loop(uCylinderLineLoop) = {lCylinder8, lCylinder5, lCylinder6, lCylinder7};
uCylinderSurf = news; Surface(uCylinderSurf) = {uCylinderLineLoop};


