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
  //   cap_radii[]
  //      Circle
  //   cap_surfaces[]
  //      Surface

  _n = # cap_nodes[];
  For _i In {0: _n - 1}
    cap_radii[_i] = newl;
    Circle(cap_radii[_i]) = {cap_nodes[_i], cap_center, cap_top};
  EndFor

  For _i In {0: _n - 2}
    _loop = newll;
    _surface = news;

    Line Loop(_loop) = {-cap_radii[_i],
                        cap_arcs[_i],
                        cap_radii[_i + 1]};
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
  //   segment_meridians[]
  //      Circle
  //   segment_surfaces[]
  //      Surface

  _n = # segment_upper_nodes[];
  For _i In {0: _n - 1}
    segment_meridians[_i] = newl;
    Circle(segment_meridians[_i]) = {segment_lower_nodes[_i],
                                     segment_center,
                                     segment_upper_nodes[_i]};
  EndFor

  For _i In {0: _n - 2}
    _loop = newll;
    _surface = news;

    Line Loop(_loop) = {segment_lower_arcs[_i],
                        segment_meridians[_i + 1],
                        -segment_upper_arcs[_i],
                        -segment_meridians[_i]};
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
  //   z, y, z, r, roi_r, roi_element_length, dihedral_angle
  //      float
  //   n_meridians
  //      int
  //   center
  //      Point
  // Returns
  // -------
  //   roi_nodes[], roi_top
  //      Point
  //   roi_arcs[], roi_radii[]
  //      Circle
  //   roi_sector_surfaces[]
  //      Surface

  h = Sqrt(r * r - roi_r * roi_r);
  For _i In {0: n_meridians-1}
    _point = newp;

    _arc = dihedral_angle * _i / (n_meridians - 1);
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
  roi_top = newp; Point(roi_top) = {x, y + r, z, roi_element_length};
  cap_top = roi_top;
  cap_nodes[] = roi_nodes[];
  cap_arcs[] = roi_arcs[];
  Call MakeSphericalCap;
  roi_sector_surfaces[] = cap_surfaces[];
  roi_radii[] = cap_radii[];
Return


Function MakeSidesOfROI
  // Arguments
  // ---------
  //   roi_upper_nodes[], roi_lower_nodes[]
  //      Point
  //   roi_upper_arcs[], roi_lower_arcs[],
  //   roi_upper_radii[], roi_lower_radii[]
  //      Circle
  //   roi_axis
  //      Line
  // Returns
  // -------
  //   roi_meridians[]
  //      Line
  //   roi_side_surfaces, roi_dihedral_surfaces[]
  //      Surface[]

  _n = # roi_upper_nodes[];
  For _i In {0: _n - 1}
    roi_meridians[_i] = newl;
    Line(roi_meridians[_i]) = {roi_lower_nodes[_i],
                               roi_upper_nodes[_i]};
  EndFor

  For _i In {0: _n - 2}
    _loop = newll;
    _surface = news;

    Line Loop(_loop) = {roi_lower_arcs[_i],
                        roi_meridians[_i + 1],
                        -roi_upper_arcs[_i],
                        -roi_meridians[_i]};
    Surface(_surface) = {_loop};

    roi_side_surfaces[_i] = _surface;
  EndFor

  For _i In {0: 1} // _n - 1: _n}
    _loop = newll;
    _surface = news;
    _idx = _i * (_n - 1);

    Line Loop(_loop) = {roi_lower_radii[_idx],
                        roi_axis,
                        -roi_upper_radii[_idx],
                        -roi_meridians[_idx]};
    Plane Surface(_surface) = {_loop};

    roi_dihedral_surfaces[_i] = _surface;
  EndFor
Return


Function MakeSphereWithROI
  // Arguments
  // ---------
  //   z, y, z, r, roi_r, element_length, roi_element_length, dihedral_angle
  //      float
  //   n_meridians
  //      int
  //   center
  //      Point
  // Returns
  // -------
  //   roi_nodes[], equatorial_nodes[], sphere_upper_pole, sphere_lower_pole
  //      Point
  //   roi_arcs[], roi_radii[], sphere_segment_meridians[],
  //   lower_hemisphere_meridians[]
  //      Circle
  //   sphere_surfaces[], roi_sector_surfaces[], surrounding_sector_surfaces[]
  //      Surface

  Call MakeCapROI;
  segment_upper_nodes[] = roi_nodes[];
  segment_upper_arcs[] = roi_arcs[];
  sphere_upper_pole = roi_top;

  For _i In {0: n_meridians-1}
     _point = newp;

     _arc = dihedral_angle * _i / (n_meridians - 1);
     Point(_point) =  {x + r * Sin(_arc),
                       y,
                       z + r * Cos(_arc),
                       element_length};

     equatorial_nodes[_i] = _point;
  EndFor


  circle_center = center;
  circle_nodes[] = equatorial_nodes[];

  Call MakeCircle;

  segment_lower_arcs[] = circle_arcs[];
  cap_nodes[] = equatorial_nodes[];
  cap_arcs[] = segment_lower_arcs[];
  sphere_lower_pole = newp; Point(sphere_lower_pole) = {x, y - r, z, element_length};
  cap_top = sphere_lower_pole;
  cap_center = center;
  Call MakeSphericalCap;
  lower_hemisphere_meridians[] = cap_radii[];

  _bottom_hemisphere_surfaces = cap_surfaces[];

  segment_center = center;
  segment_lower_nodes[] = equatorial_nodes[];
  Call MakeSphericalSegment;
  sphere_segment_meridians[] = segment_meridians[];

  surrounding_sector_surfaces = {segment_surfaces[],
                                 _bottom_hemisphere_surfaces[]};
  sphere_surfaces = {roi_sector_surfaces[],
                     surrounding_sector_surfaces[]};
Return


Function MakeSphere
  // Arguments
  // ---------
  //   z, y, z, r, element_length, dihedral_angle
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

    _arc = dihedral_angle * _i / (n_meridians - 1);
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

n_meridians = 2;
dihedral_angle = 2 * Pi / 8;
brain_r = 0.079;
scalp_r = 0.090;

brain_roi_r = 0.006;

scalp_element_length = 0.015;  // from Chaitanya's sphere_4_lowres.geo

min_sd = 0.001;
brain_roi_element_length = min_sd / 4;
scalp_roi_element_length = 0.0025;  // from Chaitanya's sphere_4_lowres.geo

x = 0.; y = 0.; z = 0.;

r = brain_r;
roi_r = brain_roi_r;
roi_element_length = brain_roi_element_length;

center = newp; Point(center) = {x, y, z, scalp_element_length};
Call MakeCapROI;

roi_upper_nodes[] = roi_nodes[];
roi_upper_arcs[] = roi_arcs[];
roi_upper_surfaces[] = roi_sector_surfaces[];
roi_upper_radii[] =  roi_radii[];
roi_upper_pole = roi_top;

r = brain_r - 2 * roi_r;
roi_r = roi_r * r / brain_r;

Call MakeCapROI;
roi_lower_nodes[] = roi_nodes[];
roi_lower_arcs[] = roi_arcs[];
roi_lower_surfaces[] = roi_sector_surfaces[];
roi_lower_radii[] =  roi_radii[];
roi_lower_pole = roi_top;

roi_axis = newl; Line(roi_axis) = {roi_lower_pole, roi_upper_pole};

Call MakeSidesOfROI;

roi_external_surfaces[] = {roi_upper_surfaces[],
                           roi_lower_surfaces[],
                           roi_side_surfaces[]};
roi_surfaces[] = {roi_external_surfaces[],
                  roi_dihedral_surfaces[]};
volume_surfaces[] = roi_surfaces[];
Call MakeVolume;
roi_volume = volume;

roi_loop = newsl; Surface Loop(roi_loop) = roi_surfaces[];

r = scalp_r;
roi_r = brain_roi_r * r / brain_r;
element_length = scalp_element_length;
roi_element_length = scalp_roi_element_length;
Call MakeSphereWithROI;
scalp_surfaces[] = sphere_surfaces[];

_top_axis = newl; Line(_top_axis) = {roi_upper_pole, sphere_upper_pole};
_middle_axis = newl; Line(_middle_axis) = {center, roi_lower_pole};
_bottom_axis = newl; Line(_bottom_axis) = {sphere_lower_pole, center};

For _i In {0: 1}
  _idx = _i * (n_meridians - 1);

  _upper_line = newl; Line(_upper_line) = {roi_upper_nodes[_idx], roi_nodes[_idx]};
  _lower_line = newl; Line(_lower_line) = {roi_lower_nodes[_idx], equatorial_nodes[_idx]};
  _equatorial_line = newl; Line(_equatorial_line) = {center, equatorial_nodes[_idx]};

  _loop = newll;
  _surface = news;
  Line Loop(_loop) = {roi_upper_radii[_idx],
                      _top_axis,
                      -roi_radii[_idx],
                      -_upper_line};
  Plane Surface(_surface) = {_loop};
  _dihedral_surfaces[_i * 4] = _surface;

  _loop = newll;
  _surface = news;
  Line Loop(_loop) = {-_lower_line,
                      roi_meridians[_idx],
                      _upper_line,
                      -sphere_segment_meridians[_idx]};
  Plane Surface(_surface) = {_loop};
  _dihedral_surfaces[_i * 4 + 1] = _surface;

  _loop = newll;
  _surface = news;
  Line Loop(_loop) = {-_equatorial_line,
                      _middle_axis,
                      -roi_lower_radii[_idx],
                      _lower_line};
  Plane Surface(_surface) = {_loop};
  _dihedral_surfaces[_i * 4 + 2] = _surface;

  _loop = newll;
  _surface = news;
  Line Loop(_loop) = {lower_hemisphere_meridians[_idx],
                      _bottom_axis,
                      _equatorial_line};
  Plane Surface(_surface) = {_loop};
  _dihedral_surfaces[_i * 4 + 3] = _surface;
EndFor

surrounding_loop = newsl;
Surface Loop(surrounding_loop) = {scalp_surfaces[],
                                  _dihedral_surfaces[],
                                  roi_external_surfaces[]};
surrounding_volume = newv; Volume(surrounding_volume) = {surrounding_loop};

Physical Volume ("brain") = {roi_volume, surrounding_volume};
Physical Surface ("scalp_surface") = scalp_surfaces[];
