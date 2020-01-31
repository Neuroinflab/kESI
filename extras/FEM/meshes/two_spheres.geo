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


Function SphericalCap
  // Arguments
  // ---------
  //   cap_center, cap_top, cap_north, cap_south, cap_west, cap_east
  //      Point
  //   cap_north_west_arc, cap_north_east_arc,
  //   cap_south_east_arc, cap_south_west_arc,
  //      Circle
  // Returns
  // -------
  //   cap_surfaces
  //      Surface[]

  cap_north_arc = newl; Circle(cap_north_arc) = {cap_north,cap_center,cap_top};
  cap_south_arc = newl; Circle(cap_south_arc) = {cap_south,cap_center,cap_top};
  cap_east_arc = newl; Circle(cap_east_arc) = {cap_east,cap_center,cap_top};
  cap_west_arc = newl; Circle(cap_west_arc) = {cap_west,cap_center,cap_top};

  cap_north_west_loop = newll; Line Loop(cap_north_west_loop) = {cap_north_west_arc,cap_west_arc,-cap_north_arc};
  cap_north_west_surface = news; Surface(cap_north_west_surface) = {cap_north_west_loop};
  cap_north_east_loop = newll; Line Loop(cap_north_east_loop) = {cap_north_east_arc,cap_east_arc,-cap_north_arc};
  cap_north_east_surface = news; Surface(cap_north_east_surface) = {cap_north_east_loop};
  cap_south_west_loop = newll; Line Loop(cap_south_west_loop) = {cap_south_west_arc,cap_west_arc,-cap_south_arc};
  cap_south_west_surface = news; Surface(cap_south_west_surface) = {cap_south_west_loop};
  cap_south_east_loop = newll; Line Loop(cap_south_east_loop) = {cap_south_east_arc,cap_east_arc,-cap_south_arc};
  cap_south_east_surface = news; Surface(cap_south_east_surface) = {cap_south_east_loop};

  cap_surfaces = {cap_north_west_surface,
                  cap_north_east_surface,
                  cap_south_west_surface,
                  cap_south_east_surface};
Return


Function SphericalSegment
  // Arguments
  // ---------
  //   segment_center,
  //   segment_upper_north, segment_upper_south, segment_upper_west, segment_upper_east
  //   segment_lower_north, segment_lower_south, segment_lower_west, segment_lower_east
  //      Point
  //   segment_upper_north_west_arc, segment_upper_north_east_arc,
  //   segment_upper_south_east_arc, segment_upper_south_west_arc,
  //   segment_lower_north_west_arc, segment_lower_north_east_arc,
  //   segment_lower_south_east_arc, segment_lower_south_west_arc,
  //      Circle
  // Returns
  // -------
  //   segment_surfaces
  //      Surface[]

  segment_north_arc = newl; Circle(segment_north_arc) = {segment_lower_north,segment_center,segment_upper_north};
  segment_south_arc = newl; Circle(segment_south_arc) = {segment_lower_south,segment_center,segment_upper_south};
  segment_east_arc = newl; Circle(segment_east_arc) = {segment_lower_east,segment_center,segment_upper_east};
  segment_west_arc = newl; Circle(segment_west_arc) = {segment_lower_west,segment_center,segment_upper_west};

  segment_north_west_loop = newll; Line Loop(segment_north_west_loop) = {segment_lower_north_west_arc,segment_west_arc,-segment_upper_north_west_arc,-segment_north_arc};
  segment_north_west_surface = news; Surface(segment_north_west_surface) = {segment_north_west_loop};
  segment_north_east_loop = newll; Line Loop(segment_north_east_loop) = {segment_lower_north_east_arc,segment_east_arc,-segment_upper_north_east_arc,-segment_north_arc};
  segment_north_east_surface = news; Surface(segment_north_east_surface) = {segment_north_east_loop};
  segment_south_west_loop = newll; Line Loop(segment_south_west_loop) = {segment_lower_south_west_arc,segment_west_arc,-segment_upper_south_west_arc,-segment_south_arc};
  segment_south_west_surface = news; Surface(segment_south_west_surface) = {segment_south_west_loop};
  segment_south_east_loop = newll; Line Loop(segment_south_east_loop) = {segment_lower_south_east_arc,segment_east_arc,-segment_upper_south_east_arc,-segment_south_arc};
  segment_south_east_surface = news; Surface(segment_south_east_surface) = {segment_south_east_loop};

  segment_surfaces = {segment_north_west_surface,
                      segment_north_east_surface,
                      segment_south_west_surface,
                      segment_south_east_surface};
Return


Function MakeCircle
  // Arguments
  // ---------
  //   circle_center, circle_north, circle_south, circle_west, circle_east 
  //      Point
  // Returns
  // -------
  //   circle_north_west_arc, circle_north_east_arc,
  //   circle_south_east_arc, circle_south_west_arc
  //      Circle
  circle_north_west_arc = newl; Circle(circle_north_west_arc) = {circle_north, circle_center, circle_west};
  circle_north_east_arc = newl; Circle(circle_north_east_arc) = {circle_north, circle_center, circle_east};
  circle_south_west_arc = newl; Circle(circle_south_west_arc) = {circle_south, circle_center, circle_west};
  circle_south_east_arc = newl; Circle(circle_south_east_arc) = {circle_south, circle_center, circle_east};
Return


Function SphereWithROI
  // Arguments
  // ---------
  //   z, y, z, r, roi_r, element_length, roi_element_length
  //      float
  //   center,
  //      Point
  // Returns
  // -------
  //   roi_north, roi_south, roi_west, roi_east
  //      Point
  //   roi_north_west_arc, roi_north_east_arc,
  //   roi_south_east_arc, roi_south_west_arc
  //      Circle
  //   roi_sector_surfaces, surrounding_sector_surfaces
  //      Surface[]

  h = Sqrt(r * r - roi_r * roi_r);
  roi_west = newp; Point(roi_west) = {x+roi_r, y+h, z, roi_element_length};
  roi_south = newp; Point(roi_south) = {x, y+h, z-roi_r, roi_element_length};
  roi_north = newp; Point(roi_north) = {x, y+h, z+roi_r, roi_element_length};
  roi_east = newp; Point(roi_east) = {x-roi_r, y+h, z, roi_element_length};

  circle_north = roi_north;
  circle_south = roi_south;
  circle_west = roi_west;
  circle_east = roi_east;
  circle_center = newp; Point(circle_center) = {x, y+h, z, roi_element_length};
  Call MakeCircle;

  roi_north_west_arc = circle_north_west_arc;
  roi_north_east_arc = circle_north_east_arc;
  roi_south_east_arc = circle_south_east_arc;
  roi_south_west_arc = circle_south_west_arc;

  cap_north_west_arc = roi_north_west_arc;
  cap_north_east_arc = roi_north_east_arc;
  cap_south_east_arc = roi_south_east_arc;
  cap_south_west_arc = roi_south_west_arc;
  cap_north = roi_north;
  cap_south = roi_south;
  cap_west = roi_west;
  cap_east = roi_east;
  cap_center = center;
  cap_top = newp; Point(cap_top) = {x, y + r, z, roi_element_length};
  Call SphericalCap;
  roi_sector_surfaces = cap_surfaces[];

  segment_lower_west = newp; Point(segment_lower_west) = {x+r, y, z, element_length};
  segment_lower_south = newp; Point(segment_lower_south) = {x, y, z-r, element_length};
  segment_lower_north = newp; Point(segment_lower_north) = {x, y, z+r, element_length};
  segment_lower_east = newp; Point(segment_lower_east) = {x-r, y, z, element_length};

  circle_north = segment_lower_north;
  circle_south = segment_lower_south;
  circle_west = segment_lower_west;
  circle_east = segment_lower_east;
  circle_center = center;

  Call MakeCircle;

  segment_lower_north_west_arc = circle_north_west_arc;
  segment_lower_north_east_arc = circle_north_east_arc;
  segment_lower_south_east_arc = circle_south_east_arc;
  segment_lower_south_west_arc = circle_south_west_arc;

  cap_north = segment_lower_north;
  cap_south = segment_lower_south;
  cap_west = segment_lower_west;
  cap_east = segment_lower_east;
  cap_north_west_arc = segment_lower_north_west_arc;
  cap_north_east_arc = segment_lower_north_east_arc;
  cap_south_east_arc = segment_lower_south_east_arc;
  cap_south_west_arc = segment_lower_south_west_arc;
  cap_top = newp; Point(cap_top) = {x, y - r, z, roi_element_length};
  Call SphericalCap;

  _bottom_hemisphere_surfaces = cap_surfaces[];

  segment_center = center;
  segment_upper_north = roi_north;
  segment_upper_south = roi_south;
  segment_upper_west = roi_west;
  segment_upper_east = roi_east;
  segment_upper_north_west_arc = roi_north_west_arc;
  segment_upper_north_east_arc = roi_north_east_arc;
  segment_upper_south_east_arc = roi_south_east_arc;
  segment_upper_south_west_arc = roi_south_west_arc;
  Call SphericalSegment;

  surrounding_sector_surfaces = {segment_surfaces[], _bottom_hemisphere_surfaces[]};
Return

x = 0.; y = 0.; z = 0.;
r = 0.079;
roi_r = 0.012;
element_length = 0.01;
min_sd = 0.001;
roi_element_length = min_sd / 4;

center = newp; Point(center) = {x, y, z, element_length};
Call SphereWithROI;
volume_surfaces = {surrounding_sector_surfaces[], roi_sector_surfaces[]};
Call MakeVolume;
