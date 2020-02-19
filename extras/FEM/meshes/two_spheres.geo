Mesh.Algorithm = 5;

Function MakeVolume
  // Arguments
  // ---------
  //   volume_surfaces
  //      Surfaces[]
  // Returns
  // -------
  //   volume
  //      Volume
  _volume_loop = newsl;
  Surface Loop(_volume_loop) = volume_surfaces[];
  volume = newv;
  Volume(volume) = _volume_loop;
Return


Function MakeSphericalCap
  // Arguments
  // ---------
  //   cap_center, cap_top, cap_nodes[]
  //      Point
  //   cap_arcs[]
  //      Circle
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


Function MakeSphericalSegment
  // Arguments
  // ---------
  //   segment_center, segment_upper_nodes[], segment_lower_nodes[]
  //      Point
  //   segment_upper_arcs[], segment_lower_arcs[]
  //      Circle
  // Returns
  // -------
  //   segment_surfaces[]
  //      Surface

  _n = # segment_upper_nodes[];
  For _i In {0: _n - 1}
    _meridians[_i] = newl;
    Circle(_meridians[_i]) = {segment_lower_nodes[_i],
                              segment_center,
                              segment_upper_nodes[_i]};
  EndFor

  For _i In {0: _n - 1}
    _loop = newll;
    _surface = news;

    Line Loop(_loop) = {segment_lower_arcs[_i],
                        _meridians[(_i + 1) % _n],
                        -segment_upper_arcs[_i],
                        -_meridians[_i]};
    Surface(_surface) = {_loop};

    segment_surfaces[_i] = _surface;
  EndFor
Return


Function MakeCircle
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


Function MakeCapROI
  // Arguments
  // ---------
  //   z, y, z, r, roi_r, roi_element_length
  //      float
  //   n_meridians
  //      int
  //   center
  //      Point
  // Returns
  // -------
  //   roi_nodes[]
  //      Point
  //   roi_arcs[]
  //      Circle
  //   roi_sector_surfaces[]
  //      Surface

  h = Sqrt(r * r - roi_r * roi_r);
  For _i In {0: n_meridians-1}
    _point = newp;

    _arc = 2 * Pi * _i / n_meridians;
    Point(_point) =  {x + roi_r * Sin(_arc),
                      y + h,
                      z + roi_r * Cos(_arc),
                      roi_element_length};

     roi_nodes[_i] = _point;
  EndFor

  circle_center = center;
  circle_nodes[] = roi_nodes[];
  Call MakeCircle;
  roi_arcs[] = circle_arcs[];

  cap_center = center;
  cap_top = newp; Point(cap_top) = {x, y + r, z, roi_element_length};
  cap_nodes[] = roi_nodes[];
  cap_arcs[] = roi_arcs[];
  Call MakeSphericalCap;
  roi_sector_surfaces[] = cap_surfaces[];
Return


Function MakeSidesOfROI
  // Arguments
  // ---------
  //   roi_upper_nodes[], roi_lower_nodes[]
  //      Point
  //   roi_upper_arcs[], roi_lower_arcs[]
  //      Circle
  // Returns
  // -------
  //   roi_side_surfaces
  //      Surface[]

  _n = # roi_upper_nodes[];
  For _i In {0: _n - 1}
    _meridians[_i] = newl;
    Line(_meridians[_i]) = {roi_lower_nodes[_i],
                            roi_upper_nodes[_i]};
  EndFor

  For _i In {0: _n - 1}
    _loop = newll;
    _surface = news;

    Line Loop(_loop) = {roi_lower_arcs[_i],
                        _meridians[(_i + 1) % _n],
                        -roi_upper_arcs[_i],
                        -_meridians[_i]};
    Surface(_surface) = {_loop};

    roi_side_surfaces[_i] = _surface;
  EndFor
Return


Function MakeSphereWithROI
  // Arguments
  // ---------
  //   z, y, z, r, roi_r, element_length, roi_element_length
  //      float
  //   n_meridians
  //      int
  //   center
  //      Point
  // Returns
  // -------
  //   roi_nodes[]
  //      Point
  //   roi_arcs[]
  //      Circle
  //   sphere_surfaces[], roi_sector_surfaces[], surrounding_sector_surfaces[]
  //      Surface

  Call MakeCapROI;
  segment_upper_nodes[] = roi_nodes[];
  segment_upper_arcs[] = roi_arcs[];

  For _i In {0: n_meridians-1}
    _point = newp;

    _arc = 2 * Pi * _i / n_meridians;
    Point(_point) =  {x + r * Sin(_arc),
                      y,
                      z + r * Cos(_arc),
                      element_length};

     segment_lower_nodes[_i] = _point;
  EndFor

  circle_center = center;
  circle_nodes[] = segment_lower_nodes[];

  Call MakeCircle;

  segment_lower_arcs[] = circle_arcs[];
  cap_nodes[] = segment_lower_nodes[];
  cap_arcs[] = segment_lower_arcs[];
  cap_top = newp; Point(cap_top) = {x, y - r, z, element_length};
  cap_center = center;
  Call MakeSphericalCap;

  _bottom_hemisphere_surfaces = cap_surfaces[];

  segment_center = center;
  Call MakeSphericalSegment;

  surrounding_sector_surfaces = {segment_surfaces[],
                                 _bottom_hemisphere_surfaces[]};
  sphere_surfaces = {roi_sector_surfaces[],
                     surrounding_sector_surfaces[]};
Return


Function MakeSphere
  // Arguments
  // ---------
  //   z, y, z, r, element_length
  //      float
  //   n_meridians
  //      int
  //   center
  //      Point
  // Returns
  // -------
  //   sphere_surfaces[]
  //      Surface

  For _i In {0: n_meridians-1}
    _point = newp;

    _arc = 2 * Pi * _i / n_meridians;
    Point(_point) =  {x + r * Sin(_arc),
                      y,
                      z + r * Cos(_arc),
                      element_length};

     circle_nodes[_i] = _point;
  EndFor

  sphere_top = newp; Point(sphere_top) = {x, y+r, z, element_length};
  sphere_bottom = newp; Point(sphere_bottom) = {x, y-r, z, element_length};

  circle_center = center;

  Call MakeCircle;
  cap_nodes[] = circle_nodes[];
  cap_arcs[] = circle_arcs[];

  cap_top = sphere_top;
  Call MakeSphericalCap;
  _upper_hemisphere_surfaces = cap_surfaces[];

  cap_top = sphere_bottom;
  Call MakeSphericalCap;
  _lower_hemisphere_surfaces = cap_surfaces[];

  sphere_surfaces[] = {_upper_hemisphere_surfaces[],
                       _lower_hemisphere_surfaces[]};
Return

n_meridians = 6;
brain_r = 0.079;
scalp_r = 0.090;

brain_roi_r = 0.006;

brain_element_length = 0.015;  // from Chaitanya's sphere_4_lowres.geo
scalp_element_length = scalp_r - brain_r;

min_sd = 0.001;
brain_roi_element_length = min_sd / 4;
scalp_roi_element_length = 0.0025;  // from Chaitanya's sphere_4_lowres.geo

x = 0.; y = 0.; z = 0.;

r = brain_r;
roi_r = brain_roi_r;
element_length = brain_element_length;
roi_element_length = brain_roi_element_length;

center = newp; Point(center) = {x, y, z, element_length};
Call MakeSphereWithROI;

roi_upper_nodes[] = roi_nodes[];
roi_upper_arcs[] = roi_arcs[];
roi_upper_surfaces[] = roi_sector_surfaces[];

r = brain_r - 2 * roi_r;
roi_r = roi_r * r / brain_r;

Call MakeCapROI;
roi_lower_nodes[] = roi_nodes[];
roi_lower_arcs[] = roi_arcs[];
roi_lower_surfaces[] = roi_sector_surfaces[];

Call MakeSidesOfROI;

volume_surfaces[] = {surrounding_sector_surfaces[],
                     roi_lower_surfaces[],
                     roi_side_surfaces[]};
Call MakeVolume;
surrounding_brain_volume = volume;

volume_surfaces[] = {roi_upper_surfaces[],
                     roi_lower_surfaces[],
                     roi_side_surfaces[]};
Call MakeVolume;
roi_volume = volume;

brain_surfaces[] = {surrounding_sector_surfaces[], roi_upper_surfaces[]};
brain_loop = newsl; Surface Loop(brain_loop) = brain_surfaces[];

r = scalp_r;
roi_r = brain_roi_r * r / brain_r;
element_length = scalp_element_length;
roi_element_length = scalp_roi_element_length;
Call MakeSphereWithROI;
scalp_surfaces[] = sphere_surfaces[];

scalp_loop = newsl; Surface Loop(scalp_loop) = scalp_surfaces[];
scalp_volume = newv; Volume(scalp_volume) = {scalp_loop, brain_loop};

Physical Volume ("brain") = {roi_volume, surrounding_brain_volume};
Physical Volume ("scalp") = scalp_volume;
Physical Surface ("brain_surface") = brain_surfaces[];
Physical Surface ("scalp_surface") = scalp_surfaces[];
