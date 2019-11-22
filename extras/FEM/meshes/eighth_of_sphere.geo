Mesh.Algorithm = 5;

Function EighthOfSphere 
  p0 = newp; Point(p0) = {x,  y,  z,  central_element_size} ;
  p1 = newp; Point(p1) = {x+surface_r,y,  z,  surface_element_size} ;
  p2 = newp; Point(p2) = {x,  y+surface_r,z,  surface_element_size} ;
  p3 = newp; Point(p3) = {x,  y,  z+surface_r,surface_element_size} ;

  p4 = newp; Point(p4) = {x+middle_r,y,  z,  middle_element_size} ;
  p5 = newp; Point(p5) = {x,  y+middle_r,z,  middle_element_size} ;
  p6 = newp; Point(p6) = {x,  y,  z+middle_r,middle_element_size} ;

  arc301 = newl; Circle(arc301) = {p3,p0,p1};
  arc102 = newl; Circle(arc102) = {p1,p0,p2};
  arc203 = newl; Circle(arc203) = {p2,p0,p3};

  arc604 = newl; Circle(arc604) = {p6,p0,p4};
  arc405 = newl; Circle(arc405) = {p4,p0,p5};
  arc506 = newl; Circle(arc506) = {p5,p0,p6};

  line04 = newl; Line(line04) = {p0, p4};
  line05 = newl; Line(line05) = {p0, p5};
  line06 = newl; Line(line06) = {p0, p6};

  line41 = newl; Line(line41) = {p4, p1};
  line52 = newl; Line(line52) = {p5, p2};
  line63 = newl; Line(line63) = {p6, p3};

  loop321 = newll; Line Loop(loop321) = {-arc102,-arc203,-arc301};
  loop654 = newll; Line Loop(loop654) = {-arc405,-arc506,-arc604};

  loop012 = newll; Line Loop(loop012) = {line04,line41,arc102,-line52,-line05};
  loop023 = newll; Line Loop(loop023) = {line05,line52,arc203,-line63,-line06};
  loop031 = newll; Line Loop(loop031) = {line06,line63,arc301,-line41,-line04};

  loop045 = newll; Line Loop(loop045) = {line04,arc405,-line05};
  loop056 = newll; Line Loop(loop056) = {line05,arc506,-line06};
  loop064 = newll; Line Loop(loop064) = {line06,arc604,-line04};

  loop4125 = newll; Line Loop(loop4125) = {line41,arc102,-line52,-arc405};
  loop5236 = newll; Line Loop(loop5236) = {line52,arc203,-line63,-arc506};
  loop6314 = newll; Line Loop(loop6314) = {line63,arc301,-line41,-arc604};

  surface321 = news;  Surface(surface321) = {loop321};
  surface654 = news;  Surface(surface654) = {loop654};
  surface012 = news;  Plane Surface(surface012) = {loop012};
  surface023 = news;  Plane Surface(surface023) = {loop023};
  surface031 = news;  Plane Surface(surface031) = {loop031};

  surface045 = news;  Plane Surface(surface045) = {loop045};
  surface056 = news;  Plane Surface(surface056) = {loop056};
  surface064 = news;  Plane Surface(surface064) = {loop064};

  surface4125 = news;  Plane Surface(surface4125) = {loop4125};
  surface5236 = news;  Plane Surface(surface5236) = {loop5236};
  surface6314 = news;  Plane Surface(surface6314) = {loop6314};

  all_loops = newsl; Surface Loop(all_loops) = {surface321,surface012,surface023,surface031};
  central_loops = newsl; Surface Loop(central_loops) = {surface654,surface045,surface056,surface064};
  external_loops = newsl; Surface Loop(external_loops) = {-surface654,surface4125,surface5236,surface6314,surface321};
Return


central_element_size = 1;
middle_element_size = 1;
surface_element_size = 10;
x = 0.; y= 0.; z=0.;
surface_r = 55.5;
// surface_r = sqrt(3) * 32 = 55.42562584220407
middle_r = 3.0;

Call EighthOfSphere ;

Physical Surface("surface_boundary") = surface321;
Physical Surface("middle") = surface654;
Physical Surface("internal_boundary") = {surface012,surface023,surface031};
whole_volume = newv;
Volume(whole_volume) = {all_loops};
central_volume = newv;
Volume(central_volume) = {central_loops};
// for some reason crashes when whole_volume is uncommented
//external_volume = newv;
//Volume(external_volume) = {external_loops};

Physical Volume("whole") = whole_volume;
Physical Volume("central") = central_volume;
//Physical Volume("external") = external_volume;

