Mesh.Algorithm = 5;


Function MakeSphericalCap
  // Arguments
  // ---------
  //   cap_center, cap_top, cap_nodes[]
  //      Point
  //   cap_arcs[]
  //      Circle
  // Alters
  // ------
  //   _cap_radii, _i, _loop, _n, _surface
  // Returns
  // -------
  //   cap_surfaces[]
  //      Surface

  _n = # cap_nodes[];
  For _i In {0: _n - 1}
    _cap_radii[_i] = newl;
    Circle(_cap_radii[_i]) = {cap_nodes[_i], cap_center, cap_top};
  EndFor

  For _i In {0: _n - 1}
    _loop = newll;
    _surface = news;

    Line Loop(_loop) = {-_cap_radii[_i],
                        cap_arcs[_i],
                        _cap_radii[(_i+1) % _n]};
    Surface(_surface) = {_loop};

    cap_surfaces[_i] = _surface;
  EndFor
Return


Function MakeSphere
  // Arguments
  // ---------
  //   x, y, z, r, element_length
  //      float
  //   n_meridians
  //      int
  //   sphere_center
  //      Point
  // Alters
  // -------
  //   cap_arcs, cap_center, cap_nodes, cap_surfaces, cap_top,
  //   circle_arcs, circle_nodes,
  //   _arc, _cap_radii, _i, _loop, _n, _point, _surface
  // Returns
  // -------
  //   sphere_surfaces[]
  //      Surface

  circle_center = sphere_center;
  Call MakeNodesOfHorizontalCircle;
  Call MakeCircleArcs;

  cap_center = sphere_center;
  cap_arcs[] = circle_arcs[];
  cap_nodes[] = circle_nodes[];

  sphere_surfaces[] = {};
  For _j In {-1: 1: 2}
    cap_top = newp;
    Point(cap_top) = {x, y, z + _j * r, element_length};
    Call MakeSphericalCap;
    sphere_surfaces[] = {sphere_surfaces[],
                         cap_surfaces[]};
  EndFor
Return


Function MakeNodesOfHorizontalCircle
  // Arguments
  // ---------
  //   z, y, z, r, element_length
  //      float
  //   n_meridians
  //      int
  // Returns
  // -------
  //   circle_nodes[]
  //      Point
  
  For _i In {0: n_meridians - 1}
    _point = newp;

    _arc = 2 * Pi * _i / n_meridians;
    Point(_point) =  {x + r * Sin(_arc),
                      y + r * Cos(_arc),
                      z,
                      element_length};

     circle_nodes[_i] = _point;
  EndFor
Return


Function MakeCircleArcs
  // Arguments
  // ---------
  //   circle_center, circle_nodes[]
  //      Point
  // Returns
  // -------
  //   circle_arcs[]
  //      Circle

  _n = # circle_nodes[];
  For _i In {0: _n - 1}
    _arc = newl;
    circle_arcs[_i] = _arc;
    Circle(_arc) = {circle_nodes[_i],
                    circle_center,
                    circle_nodes[(_i + 1) % _n]};
  EndFor
Return


Function MakeIthSphere
  // Arguments
  // ---------
  //   x, y, z, sphere_element_length_factor[], sphere_r[],
  //      float
  //   i, n_meridians
  //      int
  //   sphere_center
  //      Point
  // Alters
  // -------
  //   cap_arcs, cap_center, cap_nodes, cap_surfaces, cap_top,
  //   circle_arcs, circle_nodes,
  //   r, element_length
  //   _arc, _cap_radii, _i, _loop, _n, _point, _surface
  // Returns
  // -------
  //   sphere_surfaces[]
  //      Surface
  r = sphere_r[i];
  element_length = sphere_element_length_factor[i] * h;

  Call MakeSphere;
Return

n_meridians = 6;

h = 1;
sphere_r[] = {0.06,
              0.077,
              0.079,
              0.08,
              0.0825,
              0.085,
              0.0875,
              0.09};
sphere_element_length_factor[] = {0.15,
                                  0.01,
                                  0.005,
                                  0.005,
                                  0.025,
                                  0.0125,
                                  0.025,
                                  0.0125};

x = 0.; y = 0.; z = 0.;

sphere_center = newp;
Point(sphere_center) = {x, y, z, sphere_element_length_factor[0] * h};

i = 0;

Call MakeIthSphere;

internal_volume_loop = newsl;
Surface Loop(internal_volume_loop) = sphere_surfaces[];
volume = newv;
Volume(volume) = internal_volume_loop;

sphere_volumes[] = {volume};

For i In {1: # sphere_r[] - 1}
  Call MakeIthSphere;

  external_volume_loop = newsl;
  Surface Loop(external_volume_loop) = sphere_surfaces[];
  volume = newv;
  Volume(volume) = {external_volume_loop, internal_volume_loop};

  sphere_volumes[] = {sphere_volumes[],
                      volume};

  internal_volume_loop = external_volume_loop;
EndFor


Physical Volume ("brain") = {sphere_volumes[0],
                             sphere_volumes[1]};
Physical Volume ("csf") = {sphere_volumes[2],
                           sphere_volumes[3]};
Physical Volume ("skull") = {sphere_volumes[4],
                             sphere_volumes[5]};
Physical Volume ("scalp") = {sphere_volumes[6],
                             sphere_volumes[7]};
Physical Surface ("scalp") = sphere_surfaces[];
