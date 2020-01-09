Mesh.Algorithm = 5;

Function MakeBaseArcs
  // Arguments
  // ---------
  //   base_center, base_north, base_south, base_west, base_east
  //      Point
  // Returns
  // -------
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  base_north_west_arc = newl; Circle(base_north_west_arc) = {base_north,base_center,base_west};
  base_north_east_arc = newl; Circle(base_north_east_arc) = {base_north,base_center,base_east};
  base_south_east_arc = newl; Circle(base_south_east_arc) = {base_south,base_center,base_east};
  base_south_west_arc = newl; Circle(base_south_west_arc) = {base_south,base_center,base_west};
Return

Function SliceExternalSurface
  // Arguments
  // ---------
  //   dome_center, dome_north, dome_south, dome_west, dome_east,
  //   base_center, base_north, base_south, base_west, base_east
  //      Point
  // Returns
  // -------
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  //   slice_external_ring
  //      Surface[]
  dome_north_west_arc = newl; Circle(dome_north_west_arc) = {dome_north,dome_center,dome_west};
  dome_north_east_arc = newl; Circle(dome_north_east_arc) = {dome_north,dome_center,dome_east};
  dome_south_east_arc = newl; Circle(dome_south_east_arc) = {dome_south,dome_center,dome_east};
  dome_south_west_arc = newl; Circle(dome_south_west_arc) = {dome_south,dome_center,dome_west};

  Call MakeBaseArcs;

  slice_west_line = newl; Line(slice_west_line) = {base_west, dome_west};
  slice_north_line = newl; Line(slice_north_line) = {base_north, dome_north};
  slice_east_line = newl; Line(slice_east_line) = {base_east, dome_east};
  slice_south_line = newl; Line(slice_south_line) = {base_south, dome_south};

  slice_external_north_west_loop = newll; Line Loop(slice_external_north_west_loop) = {-dome_north_west_arc, slice_west_line, base_north_west_arc, -slice_north_line};
  slice_external_north_west_surface = news; Surface(slice_external_north_west_surface) = {slice_external_north_west_loop};
  slice_external_north_east_loop = newll; Line Loop(slice_external_north_east_loop) = {dome_north_east_arc, slice_north_line, -base_north_east_arc, -slice_east_line};
  slice_external_north_east_surface = news; Surface(slice_external_north_east_surface) = {slice_external_north_east_loop};
  slice_external_south_east_loop = newll; Line Loop(slice_external_south_east_loop) = {-dome_south_east_arc, slice_east_line, base_south_east_arc, -slice_south_line};
  slice_external_south_east_surface = news; Surface(slice_external_south_east_surface) = {slice_external_south_east_loop};
  slice_external_south_west_loop = newll; Line Loop(slice_external_south_west_loop) = {dome_south_west_arc, slice_south_line, -base_south_west_arc, -slice_west_line};
  slice_external_south_west_surface = news; Surface(slice_external_south_west_surface) = {slice_external_south_west_loop};

  slice_external_ring = {slice_external_north_west_surface,
                         slice_external_north_east_surface,
                         slice_external_south_east_surface,
                         slice_external_south_west_surface};
Return

Function SliceBaseExternalPoints
  // Arguments
  // ---------
  //   x, y, z, h, r, element_length
  //      number
  // Returns
  // -------
  //   base_north, base_south, base_east, base_west
  //      Point
  base_west = newp; Point(base_west) = {x+r,y-h,  z,  element_length} ;
  base_south = newp; Point(base_south) = {x,  y-h,  z-r,  element_length} ;
  base_north = newp; Point(base_north) = {x,  y-h,  z+r,element_length} ;
  base_east = newp; Point(base_east) = {x-r,y-h,  z,  element_length} ;
Return

Function SliceDomeExternalPoints
  // Arguments
  // ---------
  //   x, y, z, h, r, element_length
  //      number
  // Returns
  // -------
  //   dome_north, dome_south, dome_west, dome_east
  //      Point
  dome_west = newp; Point(dome_west) = {x+r,y,  z,  element_length} ;
  dome_south = newp; Point(dome_south) = {x,  y,  z-r,  element_length} ;
  dome_north = newp; Point(dome_north) = {x,  y,  z+r,element_length} ;
  dome_east = newp; Point(dome_east) = {x-r,y,  z,  element_length} ;
Return


Function SliceExternalPoints
  // Arguments
  // ---------
  //   x, y, z, h, r, element_length
  //      number
  // Returns
  // -------
  //   dome_north, dome_south, dome_west, dome_east
  //   base_north, base_south, base_east, base_west
  //      Point
  Call SliceBaseExternalPoints;
  Call SliceDomeExternalPoints;
Return

Function MakeBaseRadialLines
  // Arguments
  // ---------
  //   internal_base_north, internal_base_south,
  //   internal_base_west, internal_base_east,
  //   base_north, base_south, base_west, base_east
  //      Point
  // Returns
  // -------
  //   base_radial_west_line, base_radial_north_line,
  //   base_radial_east_line, base_radial_south_line,
  //      Line
  base_radial_west_line = newl; Line(base_radial_west_line) = {internal_base_west,base_west};
  base_radial_north_line = newl; Line(base_radial_north_line) = {internal_base_north,base_north};
  base_radial_east_line = newl; Line(base_radial_east_line) = {internal_base_east,base_east};
  base_radial_south_line = newl; Line(base_radial_south_line) = {internal_base_south,base_south};
Return

Function MakeDomeRadialLines
  // Arguments
  // ---------
  //   internal_dome_north, internal_dome_south,
  //   internal_dome_west, internal_dome_east,
  //   dome_north, dome_south, dome_west, dome_east,
  //      Point
  // Returns
  // -------
  //   dome_radial_west_line, dome_radial_north_line,
  //   dome_radial_east_line, dome_radial_south_line
  //      Line
  dome_radial_west_line = newl; Line(dome_radial_west_line) = {internal_dome_west, dome_west};
  dome_radial_north_line = newl; Line(dome_radial_north_line) = {internal_dome_north, dome_north};
  dome_radial_east_line = newl; Line(dome_radial_east_line) = {internal_dome_east, dome_east};
  dome_radial_south_line = newl; Line(dome_radial_south_line) = {internal_dome_south, dome_south};
Return

Function MakeRadialLines
  // Arguments
  // ---------
  //   internal_dome_north, internal_dome_south,
  //   internal_dome_west, internal_dome_east,
  //   dome_north, dome_south, dome_west, dome_east,
  //   internal_base_north, internal_base_south,
  //   internal_base_west, internal_base_east,
  //   base_north, base_south, base_west, base_east
  //      Point
  // Returns
  // -------
  //   base_radial_west_line, base_radial_north_line,
  //   base_radial_east_line, base_radial_south_line,
  //   dome_radial_west_line, dome_radial_north_line,
  //   dome_radial_east_line, dome_radial_south_line
  //      Line
  Call MakeBaseRadialLines;
  Call MakeDomeRadialLines;
Return


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


Function Roi
  // Arguments
  // ---------
  //   x, y, z, h, r, element_length
  //      number
  // Returns
  // -------
  //   dome_center, dome_north, dome_south, dome_west, dome_east
  //   base_center, base_north, base_south, base_east, base_west
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  //   volume
  //      Volume
  //   slice_external_ring, dome_base, slice_base
  //      Surface[]
  dome_center = newp; Point(dome_center) = {x,  y,  z,  element_length};
  base_center = newp; Point(base_center) = {x,  y-h,  z,  element_length};

  Call SliceExternalPoints;
  Call SliceExternalSurface;
  
  internal_dome_north = dome_center;
  internal_dome_south = dome_center;
  internal_dome_west = dome_center;
  internal_dome_east = dome_center;
  
  internal_base_north = base_center;
  internal_base_south = base_center;
  internal_base_west = base_center;
  internal_base_east = base_center;
  
  Call MakeRadialLines;

  dome_center_west_north_loop = newll; Line Loop(dome_center_west_north_loop) = {dome_radial_west_line, -dome_north_west_arc, -dome_radial_north_line};
  dome_center_west_north_surface = news; Plane Surface(dome_center_west_north_surface) = {dome_center_west_north_loop};
  dome_center_north_east_loop = newll; Line Loop(dome_center_north_east_loop) = {dome_radial_north_line, dome_north_east_arc, -dome_radial_east_line};
  dome_center_north_east_surface = news; Plane Surface(dome_center_north_east_surface) = {dome_center_north_east_loop};
  dome_center_east_south_loop = newll; Line Loop(dome_center_east_south_loop) = {dome_radial_east_line, -dome_south_east_arc, -dome_radial_south_line};
  dome_center_east_south_surface = news; Plane Surface(dome_center_east_south_surface) = {dome_center_east_south_loop};
  dome_center_south_west_loop = newll; Line Loop(dome_center_south_west_loop) = {dome_radial_south_line, dome_south_west_arc, -dome_radial_west_line};
  dome_center_south_west_surface = news; Plane Surface(dome_center_south_west_surface) = {dome_center_south_west_loop};

  base_center_west_north_loop = newll; Line Loop(base_center_west_north_loop) = {-base_radial_north_line, -base_north_west_arc, base_radial_west_line};
  base_center_west_north_surface = news; Plane Surface(base_center_west_north_surface) = {base_center_west_north_loop};
  base_center_north_east_loop = newll; Line Loop(base_center_north_east_loop) = {-base_radial_east_line, base_north_east_arc, base_radial_north_line};
  base_center_north_east_surface = news; Plane Surface(base_center_north_east_surface) = {base_center_north_east_loop};
  base_center_east_south_loop = newll; Line Loop(base_center_east_south_loop) = {-base_radial_south_line, -base_south_east_arc, base_radial_east_line};
  base_center_east_south_surface = news; Plane Surface(base_center_east_south_surface) = {base_center_east_south_loop};
  base_center_south_west_loop = newll; Line Loop(base_center_south_west_loop) = {-base_radial_west_line, base_south_west_arc, base_radial_south_line};
  base_center_south_west_surface = news; Plane Surface(base_center_south_west_surface) = {base_center_south_west_loop};

  dome_base = {dome_center_west_north_surface,
               dome_center_north_east_surface,
               dome_center_east_south_surface,
               dome_center_south_west_surface};
  slice_base = {base_center_north_east_surface,
                base_center_west_north_surface,
                base_center_south_west_surface,
                base_center_east_south_surface};
  volume_surfaces = {slice_external_ring[],
                     dome_base[],
                     slice_base[]};
  Call MakeVolume;
Return


Function Dome
  // Arguments
  // ---------
  //   dome_center, dome_top, dome_north, dome_south, dome_west, dome_east
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //      Circle
  // Returns
  // -------
  //   dome_surfaces
  //      Surface[]

  dome_north_arc = newl; Circle(dome_north_arc) = {dome_north,dome_center,dome_top};
  dome_south_arc = newl; Circle(dome_south_arc) = {dome_south,dome_center,dome_top};
  dome_east_arc = newl; Circle(dome_east_arc) = {dome_east,dome_center,dome_top};
  dome_west_arc = newl; Circle(dome_west_arc) = {dome_west,dome_center,dome_top};
  
  dome_north_west_loop = newll; Line Loop(dome_north_west_loop) = {dome_north_west_arc,dome_west_arc,-dome_north_arc};
  dome_north_west_surface = news; Surface(dome_north_west_surface) = {dome_north_west_loop};
  dome_north_east_loop = newll; Line Loop(dome_north_east_loop) = {dome_north_east_arc,dome_east_arc,-dome_north_arc};
  dome_north_east_surface = news; Surface(dome_north_east_surface) = {dome_north_east_loop};
  dome_south_west_loop = newll; Line Loop(dome_south_west_loop) = {dome_south_west_arc,dome_west_arc,-dome_south_arc};
  dome_south_west_surface = news; Surface(dome_south_west_surface) = {dome_south_west_loop};
  dome_south_east_loop = newll; Line Loop(dome_south_east_loop) = {dome_south_east_arc,dome_east_arc,-dome_south_arc};
  dome_south_east_surface = news; Surface(dome_south_east_surface) = {dome_south_east_loop};

  dome_surfaces = {dome_north_west_surface,
                   dome_north_east_surface,
                   dome_south_west_surface,
                   dome_south_east_surface};
Return


Function HemisphereDome
  // Arguments
  // ---------
  //   dome_center, dome_top, dome_north, dome_south, dome_west, dome_east
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //      Circle
  //   dome_base
  //      Surface[]
  // Returns
  // -------
  //   dome_surfaces
  //      Surface[]
  //   volume
  //      Volume
  Call Dome;
  volume_surfaces = {dome_surfaces[],
                     dome_base[]};
  Call MakeVolume;
Return


Function Ring
  // Arguments
  // ---------
  //   x, y, z, h, r, element_length
  //      numbers
  //   dome_center, dome_north, dome_south, dome_west, dome_east,
  //   base_center, base_north, base_south, base_west, base_east
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  //   slice_external_ring
  //      Surface[]
  // Returns
  // -------
  //   dome_north, dome_south, dome_west, dome_east,
  //   base_north, base_south, base_west, base_east
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  //   slice_internal_ring, slice_external_ring, ring_dome_base, ring_slice_base
  //      Surface[]
  //   volume
  //      Volume
  slice_internal_ring = slice_external_ring[];

  internal_dome_north_west_arc = dome_north_west_arc;
  internal_dome_north_east_arc = dome_north_east_arc;
  internal_dome_south_east_arc = dome_south_east_arc;
  internal_dome_south_west_arc = dome_south_west_arc;

  internal_base_north_west_arc = base_north_west_arc;
  internal_base_north_east_arc = base_north_east_arc;
  internal_base_south_east_arc = base_south_east_arc;
  internal_base_south_west_arc = base_south_west_arc;

  internal_dome_north = dome_north;
  internal_dome_south = dome_south;
  internal_dome_west = dome_west;
  internal_dome_east = dome_east;

  internal_base_north = base_north;
  internal_base_south = base_south;
  internal_base_west = base_west;
  internal_base_east = base_east;

  Call SliceExternalPoints;
  Call SliceExternalSurface;
  Call MakeRadialLines;

  dome_west_north_loop = newll; Line Loop(dome_west_north_loop) = {internal_dome_north_west_arc,dome_radial_west_line, -dome_north_west_arc, -dome_radial_north_line};
  dome_west_north_surface = news; Plane Surface(dome_west_north_surface) = {dome_west_north_loop};
  dome_north_east_loop = newll; Line Loop(dome_north_east_loop) = {-internal_dome_north_east_arc,dome_radial_north_line, dome_north_east_arc, -dome_radial_east_line};
  dome_north_east_surface = news; Plane Surface(dome_north_east_surface) = {dome_north_east_loop};
  dome_east_south_loop = newll; Line Loop(dome_east_south_loop) = {internal_dome_south_east_arc,dome_radial_east_line, -dome_south_east_arc, -dome_radial_south_line};
  dome_east_south_surface = news; Plane Surface(dome_east_south_surface) = {dome_east_south_loop};
  dome_south_west_loop = newll; Line Loop(dome_south_west_loop) = {-internal_dome_south_west_arc,dome_radial_south_line, dome_south_west_arc, -dome_radial_west_line};
  dome_south_west_surface = news; Plane Surface(dome_south_west_surface) = {dome_south_west_loop};

  base_west_north_loop = newll; Line Loop(base_west_north_loop) = {internal_base_north_west_arc,-base_radial_north_line, -base_north_west_arc, base_radial_west_line};
  base_west_north_surface = news; Plane Surface(base_west_north_surface) = {base_west_north_loop};
  base_north_east_loop = newll; Line Loop(base_north_east_loop) = {-internal_base_north_east_arc,-base_radial_east_line, base_north_east_arc, base_radial_north_line};
  base_north_east_surface = news; Plane Surface(base_north_east_surface) = {base_north_east_loop};
  base_east_south_loop = newll; Line Loop(base_east_south_loop) = {internal_base_south_east_arc,-base_radial_south_line, -base_south_east_arc, base_radial_east_line};
  base_east_south_surface = news; Plane Surface(base_east_south_surface) = {base_east_south_loop};
  base_south_west_loop = newll; Line Loop(base_south_west_loop) = {-internal_base_south_west_arc,-base_radial_west_line, base_south_west_arc, base_radial_south_line};
  base_south_west_surface = news; Plane Surface(base_south_west_surface) = {base_south_west_loop};
  
  ring_dome_base = {dome_west_north_surface,
                    dome_north_east_surface,
                    dome_east_south_surface,
                    dome_south_west_surface};
  ring_slice_base = {base_north_east_surface,
                     base_west_north_surface,
                     base_south_west_surface,
                     base_east_south_surface};

  volume_surfaces = {slice_internal_ring[],
                     slice_external_ring[],
                     ring_dome_base[],
                     ring_slice_base[]};
  Call MakeVolume;
Return


Function HollowDome
  // Arguments
  // ---------
  //   dome_center, dome_top, dome_north, dome_south, dome_west, dome_east
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //      Circle
  //   ring_dome_base
  //      Surface[]
  // Returns
  // -------
  //   dome_surfaces
  //      Surface[]
  //   volume
  //      Volume
  internal_dome_surfaces = dome_surfaces[];
  Call Dome;
  volume_surfaces = {dome_surfaces[],
                     ring_dome_base[],
                     internal_dome_surfaces[]};
  Call MakeVolume;
Return


Function BaseHollowDome
  // Arguments
  // ---------
  //   x, y, z, h, r, element_length
  //      numbers
  //   base_center, base_north, base_south, base_west, base_east
  //      Point
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  // Returns
  // -------
  //   dome_north, dome_south, dome_west, dome_east,
  //   base_north, base_south, base_west, base_east
  //      Point
  //   dome_north_west_arc, dome_north_east_arc,
  //   dome_south_east_arc, dome_south_west_arc,
  //   base_north_west_arc, base_north_east_arc,
  //   base_south_east_arc, base_south_west_arc
  //      Circle
  //   ring_dome_base
  //      Surface[]
  //   volume
  //      Volume
  slice_internal_ring = slice_external_ring[];

  internal_base_north_west_arc = base_north_west_arc;
  internal_base_north_east_arc = base_north_east_arc;
  internal_base_south_east_arc = base_south_east_arc;
  internal_base_south_west_arc = base_south_west_arc;

  internal_base_north = base_north;
  internal_base_south = base_south;
  internal_base_west = base_west;
  internal_base_east = base_east;

  Call SliceBaseExternalPoints;
  Call MakeBaseRadialLines;
  Call MakeBaseArcs;

  base_west_north_loop = newll; Line Loop(base_west_north_loop) = {internal_base_north_west_arc,-base_radial_north_line, -base_north_west_arc, base_radial_west_line};
  base_west_north_surface = news; Plane Surface(base_west_north_surface) = {base_west_north_loop};
  base_north_east_loop = newll; Line Loop(base_north_east_loop) = {-internal_base_north_east_arc,-base_radial_east_line, base_north_east_arc, base_radial_north_line};
  base_north_east_surface = news; Plane Surface(base_north_east_surface) = {base_north_east_loop};
  base_east_south_loop = newll; Line Loop(base_east_south_loop) = {internal_base_south_east_arc,-base_radial_south_line, -base_south_east_arc, base_radial_east_line};
  base_east_south_surface = news; Plane Surface(base_east_south_surface) = {base_east_south_loop};
  base_south_west_loop = newll; Line Loop(base_south_west_loop) = {-internal_base_south_west_arc,-base_radial_west_line, base_south_west_arc, base_radial_south_line};
  base_south_west_surface = news; Plane Surface(base_south_west_surface) = {base_south_west_loop};

  ring_dome_base = {base_north_east_surface,
                    base_west_north_surface,
                    base_south_west_surface,
                    base_east_south_surface};

  dome_north = base_north;
  dome_south = base_south;
  dome_west = base_west;
  dome_east = base_east;

  dome_north_west_arc = base_north_west_arc;
  dome_north_east_arc = base_north_east_arc;
  dome_south_east_arc = base_south_east_arc;
  dome_south_west_arc = base_south_west_arc;

  Call HollowDome;
Return


h = 0.0003;
x = 0.; y = h; z = 0.;
r = h;
element_length = 3 * 0.03125*h;
Call Roi ;
slice_volumes = {volume};

//Physical Volume ("ROI") = volume;


r = 0.003;
element_length = 0.4 * 0.03125*r;

Call Ring;
dome_base = {dome_base[], ring_dome_base[]};
slice_base = {slice_base[], ring_slice_base[]};
slice_volumes = {slice_volumes[], volume};


dome_top = newp; Point(dome_top) = {x,  y+r,z, element_length};
Call HemisphereDome ;
saline_volumes = {volume};
dome_base = {dome_base[], slice_external_ring[]};
dome_surfaces = {dome_surfaces[], slice_external_ring[]};
dome_center = base_center;

//r = 0.03; element_length = 0.3*r;
//dome_top = newp; Point(dome_top) = {x,  y+r-h,z, element_length};
//Call BaseHollowDome;
//dome_base = {dome_base[], ring_dome_base[]};
//saline_volumes = {saline_volumes[], volume};


Physical Volume ("slice") = slice_volumes[];
Physical Volume ("saline") = saline_volumes[];
Physical Surface("dome") = dome_surfaces[];
//Physical Surface("slice_surface") = dome_base[];
//Physical Surface("slice_base") = slice_base[];
