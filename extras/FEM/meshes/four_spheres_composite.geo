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


Function MakeCapROI
  // Arguments
  // ---------
  //   z, y, z, r, roi_r, roi_element_length
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
  //   roi_sector_surfaces
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
  circle_center = center;
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
Return


Function RingOfROI
  // Arguments
  // ---------
  //   roi_top_north, roi_top_south, roi_top_west, roi_top_east,
  //   roi_bottom_north, roi_bottom_south, roi_bottom_west, roi_bottom_east
  //      Point
  //   roi_north_west_upper_arc, roi_north_east_upper_arc,
  //   roi_south_east_upper_arc, roi_south_west_upper_arc,
  //   roi_north_west_lower_arc, roi_north_east_lower_arc,
  //   roi_south_east_lower_arc, roi_south_west_lower_arc
  //      Circle
  // Returns
  // -------
  //   roi_ring_surfaces
  //      Surface[]

  roi_north_line = newl; Line(roi_north_line) = {roi_bottom_north, roi_top_north};
  roi_south_line = newl; Line(roi_south_line) = {roi_bottom_south, roi_top_south};
  roi_east_line = newl; Line(roi_east_line) = {roi_bottom_east, roi_top_east};
  roi_west_line = newl; Line(roi_west_line) = {roi_bottom_west, roi_top_west};

  roi_north_west_loop = newll;
  Line Loop(roi_north_west_loop) = {roi_north_line,roi_north_west_upper_arc,-roi_west_line,-roi_north_west_lower_arc};
  roi_north_east_loop = newll;
  Line Loop(roi_north_east_loop) = {roi_north_line,roi_north_east_upper_arc,-roi_east_line,-roi_north_east_lower_arc};
  roi_south_west_loop = newll;
  Line Loop(roi_south_west_loop) = {roi_south_line,roi_south_west_upper_arc,-roi_west_line,-roi_south_west_lower_arc};
  roi_south_east_loop = newll;
  Line Loop(roi_south_east_loop) = {roi_south_line,roi_south_east_upper_arc,-roi_east_line,-roi_south_east_lower_arc};

  roi_north_west_surface = news;
  Surface(roi_north_west_surface) = {roi_north_west_loop};
  roi_north_east_surface = news;
  Surface(roi_north_east_surface) = {roi_north_east_loop};
  roi_south_west_surface = news;
  Surface(roi_south_west_surface) = {roi_south_west_loop};
  roi_south_east_surface = news;
  Surface(roi_south_east_surface) = {roi_south_east_loop};

  roi_ring_surfaces = {roi_north_west_surface, roi_north_east_surface, roi_south_west_surface, roi_south_east_surface};
Return


Function MakeSphereWithROI
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
  //   sphere_surfaces, roi_sector_surfaces, surrounding_sector_surfaces
  //      Surface[]

  Call MakeCapROI;

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
  cap_top = newp; Point(cap_top) = {x, y - r, z, element_length};
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
  //   center
  //      Point
  // Returns
  // -------
  //   sphere_surfaces
  //      Surface[]

  sphere_west = newp; Point(sphere_west) = {x+r, y, z, element_length};
  sphere_south = newp; Point(sphere_south) = {x, y, z-r, element_length};
  sphere_north = newp; Point(sphere_north) = {x, y, z+r, element_length};
  sphere_east = newp; Point(sphere_east) = {x-r, y, z, element_length};
  sphere_top = newp; Point(sphere_top) = {x, y+r, z, element_length};
  sphere_bottom = newp; Point(sphere_bottom) = {x, y-r, z, element_length};

  circle_north = sphere_north;
  circle_south = sphere_south;
  circle_west = sphere_west;
  circle_east = sphere_east;
  circle_center = center;

  Call MakeCircle;
  cap_north_west_arc = circle_north_west_arc;
  cap_north_east_arc = circle_north_east_arc;
  cap_south_east_arc = circle_south_east_arc;
  cap_south_west_arc = circle_south_west_arc;

  cap_north = sphere_north;
  cap_south = sphere_south;
  cap_west = sphere_west;
  cap_east = sphere_east;

  cap_top = sphere_top;
  Call SphericalCap;
  _upper_hemisphere_surfaces = cap_surfaces[];

  cap_top = sphere_bottom;
  Call SphericalCap;
  _lower_hemisphere_surfaces = cap_surfaces[];

  sphere_surfaces = {_upper_hemisphere_surfaces[], _lower_hemisphere_surfaces[]};
Return

brain_r = 0.079;
csf_r   = 0.080;
skull_r = 0.085;
scalp_r = 0.090;

brain_roi_r = 0.006;

brain_element_length = 0.015;  // from Chaitanya's sphere_4_lowres.geo
csf_element_length   = csf_r - brain_r;
skull_element_length = skull_r - csf_r;
scalp_element_length = scalp_r - skull_r;

min_sd = 0.001;
brain_roi_element_length = min_sd / 4;
csf_roi_element_length   = 0.0050;  // from Chaitanya's sphere_4_lowres.geo
skull_roi_element_length = 0.0025;  // from Chaitanya's sphere_4_lowres.geo
scalp_roi_element_length = 0.0025;  // from Chaitanya's sphere_4_lowres.geo

x = 0.; y = 0.; z = 0.;

r = brain_r;
roi_r = brain_roi_r;
element_length = csf_element_length;
roi_element_length = brain_roi_element_length;

center = newp; Point(center) = {x, y, z, element_length};
Call MakeSphereWithROI;

roi_top_north = roi_north;
roi_top_south = roi_south;
roi_top_west = roi_west;
roi_top_east = roi_east;
roi_north_west_upper_arc = roi_north_west_arc;
roi_north_east_upper_arc = roi_north_east_arc;
roi_south_east_upper_arc = roi_south_east_arc;
roi_south_west_upper_arc = roi_south_west_arc;
roi_sector_upper_surfaces = roi_sector_surfaces[];
external_surrounding_sector_surfaces = surrounding_sector_surfaces[];
cortex_surfaces = sphere_surfaces[];
cortex_loop = newsl; Surface Loop(cortex_loop) = cortex_surfaces[];

r = brain_r - 2 * roi_r;
roi_r = roi_r * r / brain_r;
element_length = brain_element_length;

Call MakeSphereWithROI;
roi_bottom_north = roi_north;
roi_bottom_south = roi_south;
roi_bottom_west = roi_west;
roi_bottom_east = roi_east;
roi_north_west_lower_arc = roi_north_west_arc;
roi_north_east_lower_arc = roi_north_east_arc;
roi_south_east_lower_arc = roi_south_east_arc;
roi_south_west_lower_arc = roi_south_west_arc;
roi_sector_lower_surfaces = roi_sector_surfaces[];
internal_surrounding_sector_surfaces = surrounding_sector_surfaces[];

volume_surfaces = sphere_surfaces[];
Call MakeVolume;
subcortex_volume = volume;

Call RingOfROI;

volume_surfaces = {external_surrounding_sector_surfaces[],
                   internal_surrounding_sector_surfaces[],
                   roi_ring_surfaces[]};
Call MakeVolume;
surrounding_cortex_volume = volume;

volume_surfaces = {roi_sector_upper_surfaces[],
                   roi_sector_lower_surfaces[],
                   roi_ring_surfaces[]};
Call MakeVolume;
roi_volume = volume;


r = csf_r;
roi_r = brain_roi_r * r / brain_r;
element_length = csf_element_length;
roi_element_length = csf_roi_element_length;
Call MakeSphereWithROI;

csf_loop = newsl; Surface Loop(csf_loop) = sphere_surfaces[];
csf_volume = newv; Volume(csf_volume) = {csf_loop, cortex_loop};

r = skull_r;
roi_r = brain_roi_r * r / brain_r;
element_length = skull_element_length;
roi_element_length = skull_roi_element_length;
Call MakeSphereWithROI;

skull_loop = newsl; Surface Loop(skull_loop) = sphere_surfaces[];
skull_volume = newv; Volume(skull_volume) = {skull_loop, csf_loop};


r = scalp_r;
roi_r = brain_roi_r * r / brain_r;
element_length = scalp_element_length;
roi_element_length = scalp_roi_element_length;
Call MakeSphereWithROI;

scalp_surfaces = sphere_surfaces[];
scalp_loop = newsl; Surface Loop(scalp_loop) = scalp_surfaces[];
scalp_volume = newv; Volume(scalp_volume) = {scalp_loop, skull_loop};

Physical Volume ("brain") = {roi_volume,
                             surrounding_cortex_volume,
                             subcortex_volume};
Physical Volume ("csf") = csf_volume;
Physical Volume ("skull") = skull_volume;
Physical Volume ("scalp") = scalp_volume;
Physical Surface ("brain_surface") = cortex_surfaces[];
Physical Surface ("scalp_surface") = scalp_surfaces[];
