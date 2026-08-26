// Gmsh project created on Fri May 10 17:43:53 2024
Mesh.Algorithm = 5;
Mesh.MshFileVersion = 2.2;

General.NumThreads = 16;


Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MeshSizeFactor = 1;
Mesh.MeshSizeMin = 0.0;
Mesh.MeshSizeMax  = 0.005;
//Automatically compute mesh element sizes from curvature, using the value as the target number of elements per 2 * Pi radians
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeFromPoints = 1;

//Mesh.MinimumLineNodes = 2;
//Mesh.MinimumCircleNodes = 7;
Mesh.MinimumCircleNodes = 128;
Mesh.MinimumCurveNodes = 128;

//Mesh.CharacteristicLengthFromCurvature = 1;
//Mesh.MinimumElementsPerTwoPi = 10;

SetFactory("OpenCASCADE");
//+
Sphere(1) = {0, 0, 0, 0.079, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(2) = {0, 0, 0, 0.082, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(3) = {0, 0, 0, 0.086, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(4) = {0, 0, 0, 0.09, -Pi/2, Pi/2, 2*Pi};

Box(5) = {-0.15, -0.15, -0.1, 0.3, 0.3, 0.3};

v() = BooleanFragments {Volume{5}; Delete;}{Volume{1,2,3,4}; Delete;};//+
Coherence;

Physical Surface(1) = {6};
//+
Physical Volume(1) = {1};
Physical Volume(2) = {3};
Physical Volume(3) = {4};
Physical Volume(4) = {5};
Physical Volume(5) = {2};