// Single slice
// Saline to Layer I
p1 = newp; Point(p1) = {slice_bottom, cortex_width/2, -cortex_length/2, meshCortex};
p2 = newp; Point(p2) = {slice_thickness/2, cortex_width/2, -cortex_length/2, meshCortex};
p3 = newp; Point(p3) = {slice_thickness/2, -cortex_width/2, -cortex_length/2, meshCortex};
p4 = newp; Point(p4) = {slice_bottom,-cortex_width/2, -cortex_length/2, meshCortex};

p5 = newp; Point(p5) = {slice_bottom, cortex_width/2, cortex_length/2, meshCortex};
p6 = newp; Point(p6) = {slice_thickness/2, cortex_width/2, cortex_length/2, meshCortex};
p7 = newp; Point(p7) = {slice_thickness/2, -cortex_width/2, cortex_length/2, meshCortex};
p8 = newp; Point(p8) = {slice_bottom, -cortex_width/2, cortex_length/2, meshCortex};

//Saline_layer beneath - electrode plane
p1b = newp; Point(p1b) = {-slice_thickness/2, cortex_width/2, -cortex_length/2, meshCortexLower};
p4b = newp; Point(p4b) = {-slice_thickness/2,-cortex_width/2, -cortex_length/2, meshCortexLower};
p5b = newp; Point(p5b) = {-slice_thickness/2, cortex_width/2, cortex_length/2, meshCortexLower};
p8b = newp; Point(p8b) = {-slice_thickness/2, -cortex_width/2, cortex_length/2, meshCortexLower};

l5b8b = newl; Line(l5b8b) = {p5b, p8b};
l4b1b = newl; Line(l4b1b) = {p4b, p1b};

l1b1 = newl; Line(l1b1) = {p1b, p1};// surface facing -Z
l44b = newl; Line(l44b) = {p4, p4b};
l55b = newl; Line(l55b) = {p5, p5b};// surface facing +Z
l8b8 = newl; Line(l8b8) = {p8b, p8};     

l84 = newl; Line(l84) = {p8, p4};
l15 = newl; Line(l15) = {p1, p5};

l41 = newl; Line(l41) = {p4, p1};
l12 = newl; Line(l12) = {p1, p2};
l23 = newl; Line(l23) = {p2, p3};
l34 = newl; Line(l34) = {p3, p4};
llS1 = newll; Line Loop(llS1) = {l1b1, l12, l23, l34, l44b, l4b1b};
sS1 = news; Plane Surface(sS1) = {llS1}; // surface facing -Z

l65 = newl; Line(l65) = {p6, p5};
l58 = newl; Line(l58) = {p5, p8};
l87 = newl; Line(l87) = {p8, p7};
l76 = newl; Line(l76) = {p7, p6};
llS3 = newll; Line Loop(llS3) = {l65, l55b, l5b8b, l8b8, l87, l76};
sS3 = news; Plane Surface(sS3) = {llS3}; // surface facing +Z

//top surface - top of slice
l62 = newl; Line(l62) = {p6, p2};
l37 = newl; Line(l37) = {p3, p7};
llS5 = newll; Line Loop(llS5) = {-l62, -l76, -l37, -l23};
sS5 = news; Plane Surface(sS5) = {llS5}; // surface facing +X

// bottom surface - electrode plane
l8b4b = newl; Line(l8b4b) = {p8b, p4b};
l1b5b = newl; Line(l1b5b) = {p1b, p5b};

llS6 = newll; Line Loop(llS6) = {-l1b5b, -l5b8b, -l8b4b, -l4b1b};
sS6 = news; Plane Surface(sS6) = {llS6}; // surface facing -X

llS2 = newll; Line Loop(llS2) = {-l1b1, -l12, l62, -l65, -l55b, l1b5b};
sS2 = news; Plane Surface(sS2) = {llS2}; // surface facing +Y

llS4 = newll; Line Loop(llS4) = {l8b4b, -l44b, -l34, l37, -l87, -l8b8};
sS4 = news; Plane Surface(sS4) = {llS4}; // surface facing -Y 

slS = newsl; Surface Loop(slS) = {sS1, sS2, sS3, sS4, sS5, sS6};
vSlice = newv; Volume(vSlice) = {slS}; // volume of slice
