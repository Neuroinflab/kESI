// Gmsh project created on Fri May 10 17:43:53 2024
Mesh.Algorithm = 5;
Mesh.MshFileVersion = 2.2;

General.NumThreads = 16;


SetFactory("OpenCASCADE");
//+
Sphere(1) = {0, 0, 0, 0.079, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(2) = {0, 0, 0, 0.082, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(3) = {0, 0, 0, 0.086, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(4) = {0, 0, 0, 0.09, -Pi/2, Pi/2, 2*Pi};

Box(10) = {-100, -100, -100, 200, 200, 200};


BooleanDifference(20) = {Volume{10}; Delete;}{Volume{1,2,3,4};};
BooleanDifference(7) = {Volume{4}; Delete;}{Volume{1,2,3};};
BooleanDifference(6) = {Volume{3}; Delete;}{Volume{1,2};};
BooleanDifference(5) = {Volume{2}; Delete;}{Volume{1};};

//+
Disk(30) = {-0, 0, -0.078, 0.002, 0.002};


v() = BooleanFragments {Volume{1}; Delete;}{Surface{30}; Delete;};//+

Physical Surface(1) = {5, 6, 7, 8, 9, 10, 30};
Physical Volume(1) = {1};
Physical Volume(2) = {5};
Physical Volume(3) = {6};
Physical Volume(4) = {7};
Physical Volume(5) = {20};


Field[1] = Box;
Field[1].VIn = 0.005; // Mesh size inside the cube
Field[1].VOut = 2; // Mesh size outside the cube
Field[1].XMin = -0.15; // X min of the cube
Field[1].XMax = 0.15; // X max of the cube
Field[1].YMin = -0.15; // Y min of the cube
Field[1].YMax = 0.15; // Y max of the cube
Field[1].ZMin = -0.15; // Z min of the cube
Field[1].ZMax = 0.15; // Z max of the cube


Field[2] = Box;
Field[2].VIn = 0.02; // Mesh size inside the cube
Field[2].VOut = 2; // Mesh size outside the cube
Field[2].XMin = -0.5; // X min of the cube
Field[2].XMax = 0.5; // X max of the cube
Field[2].YMin = -0.5; // Y min of the cube
Field[2].YMax = 0.5; // Y max of the cube
Field[2].ZMin = -0.5; // Z min of the cube
Field[2].ZMax = 0.5; // Z max of the cube

Field[4] = Min;  // combine mesh size fields by using minimum criteria
Field[4].FieldsList = {1, 2};

// Apply the field
Background Field = 4;


