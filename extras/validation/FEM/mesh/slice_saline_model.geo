Mesh.Algorithm = 5;

Function SliceSaline3d

  pcenterdome = newp; Point(pcenterdome) = {x,  y,  z,  lcar1} ;
  pdomepeak = newp; Point(pdomepeak) = {x,  y+r,z,  lcar3} ;
  pdomebase1 = newp; Point(pdomebase1) = {x+r,y,  z,  lcar3} ;
  pdomebase2 = newp; Point(pdomebase2) = {x,  y,  z+r,lcar3} ;
  pdomebase3 = newp; Point(pdomebase3) = {x-r,y,  z,  lcar3} ;
  pdomebase4 = newp; Point(pdomebase4) = {x,  y,  z-r,lcar3} ;
  pcenterslice = newp; Point(pcenterslice) = {x,  y-h,  z,  lcar1} ;
  pslicebase1 = newp; Point(pslicebase1) = {x+r,y-h,  z,  lcar3} ;
  pslicebase4 = newp; Point(pslicebase4) = {x,  y-h,  z-r,  lcar3} ;
  pslicebase2 = newp; Point(pslicebase2) = {x,  y-h,  z+r,lcar3} ;
  pslicebase3 = newp; Point(pslicebase3) = {x-r,y-h,  z,  lcar3} ;
  proislice1 = newp; Point(proislice1) = {x+rs,y-h,  z,  lcar1} ;
  proislice4 = newp; Point(proislice4) = {x,  y-h,  z-rs,  lcar1} ;
  proislice2 = newp; Point(proislice2) = {x,  y-h,  z+rs,lcar1} ;
  proislice3 = newp; Point(proislice3) = {x-rs,y-h,  z,  lcar1} ;
  proidome1 = newp; Point(proidome1) = {x+rs,y,  z,  lcar1} ;
  proidome4 = newp; Point(proidome4) = {x,  y,  z-rs,  lcar1} ;
  proidome2 = newp; Point(proidome2) = {x,  y,  z+rs,lcar1} ;
  proidome3 = newp; Point(proidome3) = {x-rs,y,  z,  lcar1} ;

  croidome1 = newreg; Circle(croidome1) = {proidome1,pcenterdome,proidome2};
  croidome2 = newreg; Circle(croidome2) = {proidome2,pcenterdome,proidome3};
  croidome3 = newreg; Circle(croidome3) = {proidome3,pcenterdome,proidome4};
  croidome4 = newreg; Circle(croidome4) = {proidome4,pcenterdome,proidome1};
  cdome1 = newreg; Circle(cdome1) = {pdomebase1,pcenterdome,pdomepeak};
  cdome2 = newreg; Circle(cdome2) = {pdomepeak,pcenterdome,pdomebase2};
  cdome3 = newreg; Circle(cdome3) = {pdomepeak,pcenterdome,pdomebase3};
  cdome4 = newreg; Circle(cdome4) = {pdomebase4,pcenterdome,pdomepeak};
  cslice1 = newreg; Circle(cslice1) = {pslicebase2,pcenterslice,pslicebase1};
  cslice2 = newreg; Circle(cslice2) = {pslicebase3,pcenterslice,pslicebase2};
  cslice3 = newreg; Circle(cslice3) = {pslicebase4,pcenterslice,pslicebase3};
  cslice4 = newreg; Circle(cslice4) = {pslicebase4,pcenterslice,pslicebase1};
  cdomebase1 = newreg; Circle(cdomebase1) = {pdomebase2,pcenterdome,pdomebase1};
  cdomebase2 = newreg; Circle(cdomebase2) = {pdomebase3,pcenterdome,pdomebase2};
  cdomebase3 = newreg; Circle(cdomebase3) = {pdomebase4,pcenterdome,pdomebase3};
  cdomebase4 = newreg; Circle(cdomebase4) = {pdomebase1,pcenterdome,pdomebase4};
  croislice1 = newreg; Circle(croislice1) = {proislice1,pcenterslice,proislice2};
  croislice2 = newreg; Circle(croislice2) = {proislice2,pcenterslice,proislice3};
  croislice3 = newreg; Circle(croislice3) = {proislice3,pcenterslice,proislice4};
  croislice4 = newreg; Circle(croislice4) = {proislice4,pcenterslice,proislice1};

  lroidome1 = newl; Line(lroidome1) = {pcenterdome, proidome1};
  lroidome2 = newl; Line(lroidome2) = {pcenterdome, proidome2};
  lroidome3 = newl; Line(lroidome3) = {pcenterdome, proidome3};
  lroidome4 = newl; Line(lroidome4) = {pcenterdome, proidome4};
  lsliceds1 = newl; Line(lsliceds1) = {pdomebase1, pslicebase1};
  lsliceds2 = newl; Line(lsliceds2) = {pdomebase2, pslicebase2};
  lsliceds3 = newl; Line(lsliceds3) = {pdomebase3, pslicebase3};
  lsliceds4 = newl; Line(lsliceds4) = {pdomebase4, pslicebase4};
  lslicebase1 = newl; Line(lslicebase1) = {proislice1, pslicebase1};
  lslicebase2 = newl; Line(lslicebase2) = {proislice2, pslicebase2};
  lslicebase3 = newl; Line(lslicebase3) = {proislice3, pslicebase3};
  lslicebase4 = newl; Line(lslicebase4) = {proislice4, pslicebase4};
  lroislice1 = newl; Line(lroislice1) = {proislice1, pcenterslice};
  lroislice2 = newl; Line(lroislice2) = {proislice2, pcenterslice};
  lroislice3 = newl; Line(lroislice3) = {proislice3, pcenterslice};
  lroislice4 = newl; Line(lroislice4) = {proislice4, pcenterslice};  
  ldomebase1 = newl; Line(ldomebase1) = {pdomebase1, proidome1};
  ldomebase2 = newl; Line(ldomebase2) = {pdomebase2, proidome2};
  ldomebase3 = newl; Line(ldomebase3) = {pdomebase3, proidome3};
  ldomebase4 = newl; Line(ldomebase4) = {pdomebase4, proidome4};
  lroids1 = newl; Line(lroids1) = {proislice1, proidome1};
  lroids2 = newl; Line(lroids2) = {proislice2, proidome2};
  lroids3 = newl; Line(lroids3) = {proislice3, proidome3};
  lroids4 = newl; Line(lroids4) = {proislice4, proidome4};

  lldome1 = newll; Line Loop(lldome1) = {cdome1,cdome2,cdomebase1};   Ruled Surface(newreg) = {lldome1};
  lldome2 = newll; Line Loop(lldome2) = {-cdome2,cdome3,cdomebase2};  Ruled Surface(newreg) = {lldome2};
  lldome3 = newll; Line Loop(lldome3) = {-cdome3,-cdome4,cdomebase3};  Ruled Surface(newreg) = {lldome3};
  lldome4 = newll; Line Loop(lldome4) = {cdome4,-cdome1,cdomebase4};   Ruled Surface(newreg) = {lldome4};
  llroidome1 = newll; Line Loop(llroidome1) = {lroidome1, croidome1, -lroidome2}; Plane Surface(newreg) = {llroidome1};
  llroidome2 = newll; Line Loop(llroidome2) = {lroidome2, croidome2, -lroidome3}; Plane Surface(newreg) = {llroidome2};
  llroidome3 = newll; Line Loop(llroidome3) = {lroidome3, croidome3, -lroidome4}; Plane Surface(newreg) = {llroidome3};
  llroidome4 = newll; Line Loop(llroidome4) = {lroidome4, croidome4, -lroidome1}; Plane Surface(newreg) = {llroidome4};
  lldomebase1 = newll; Line Loop(lldomebase1) = {ldomebase1, cdomebase1, -ldomebase2,croidome1};   Ruled Surface(newreg) = {lldomebase1};
  lldomebase2 = newll; Line Loop(lldomebase2) = {croidome2, ldomebase2, cdomebase2, -ldomebase3};   Ruled Surface(newreg) = {lldomebase2};
  lldomebase3 = newll; Line Loop(lldomebase3) = {croidome3, ldomebase3, cdomebase3, -ldomebase4};  Ruled Surface(newreg) = {lldomebase3};
  lldomebase4 = newll; Line Loop(lldomebase4) = {croidome4, ldomebase4, cdomebase4, -ldomebase1};  Ruled Surface(newreg) = {lldomebase4};
  llroids1 = newll; Line Loop(llroids1) = {croidome1, lroids1, -croislice1, -lroids2};   Ruled Surface(newreg) = {llroids1};
  llroids2 = newll; Line Loop(llroids2) = {croidome2, lroids2, -croislice2, -lroids3};   Ruled Surface(newreg) = {llroids2};
  llroids3 = newll; Line Loop(llroids3) = {croidome3, lroids3, -croislice3, -lroids4};  Ruled Surface(newreg) = {llroids3};
  llroids4 = newll; Line Loop(llroids4) = {croidome4, lroids4, -croislice4, -lroids1};  Ruled Surface(newreg) = {llroids4};
  llroislice1 = newll; Line Loop(llroislice1) = {lroislice2, croislice1, -lroislice1}; Ruled Surface(newreg) = {llroislice1};
  llroislice2 = newll; Line Loop(llroislice2) = {lroislice3, croislice2, -lroislice2}; Ruled Surface(newreg) = {llroislice2};
  llroislice3 = newll; Line Loop(llroislice3) = {lroislice4, croislice3, -lroislice3}; Ruled Surface(newreg) = {llroislice3};
  llroislice4 = newll; Line Loop(llroislice4) = {lroislice1, croislice4, -lroislice4}; Ruled Surface(newreg) = {llroislice4};
  llslicebase1 = newll; Line Loop(llslicebase1) = {croislice1, -lslicebase1, cslice1, lslicebase2};   Ruled Surface(newreg) = {llslicebase1};
  llslicebase2 = newll; Line Loop(llslicebase2) = {croislice2, -lslicebase2, cslice2, lslicebase3};   Ruled Surface(newreg) = {llslicebase2};
  llslicebase3 = newll; Line Loop(llslicebase3) = {croislice3, -lslicebase3, cslice3, lslicebase4};  Ruled Surface(newreg) = {llslicebase3};
  llslicebase4 = newll; Line Loop(llslicebase4) = {croislice4, -lslicebase4, -cslice4, lslicebase1};  Ruled Surface(newreg) = {llslicebase4};
  llsliceds4 = newll; Line Loop(llsliceds4) = {cdomebase4, lsliceds4, cslice4, -lsliceds1};   Ruled Surface(newreg) = {llsliceds4};
  llsliceds1 = newll; Line Loop(llsliceds1) = {cdomebase1, lsliceds1, -cslice1, -lsliceds2};   Ruled Surface(newreg) = {llsliceds1};
  llsliceds2 = newll; Line Loop(llsliceds2) = {cdomebase2, lsliceds2, -cslice2, -lsliceds3};  Ruled Surface(newreg) = {llsliceds2};
  llsliceds3 = newll; Line Loop(llsliceds3) = {cdomebase3, lsliceds3, -cslice3, -lsliceds4};  Ruled Surface(newreg) = {llsliceds3};


  elem_dome = {lldome3+1,lldome2+1,lldome1+1,lldome4+1,llsliceds4+1,llsliceds1+1,llsliceds2+1,llsliceds3+1};

  elem_model_base = {llroislice2+1,llroislice1+1,llroislice4+1,llroislice3+1,llslicebase1+1,llslicebase2+1,llslicebase3+1,llslicebase4+1};

  elem_saline = newreg;
  salinesurface = {lldome1+1,lldome4+1,lldome2+1,lldome3+1,llroidome1+1,llroidome2+1,llroidome3+1,llroidome4+1,lldomebase1+1,lldomebase2+1,lldomebase3+1,lldomebase4+1};
  Surface Loop(elem_saline) = {lldome1+1,lldome4+1,lldome2+1,lldome3+1,llroidome1+1,llroidome2+1,llroidome3+1,llroidome4+1,lldomebase1+1,lldomebase2+1,lldomebase3+1,lldomebase4+1};
  //Surface Loop(elem_saline) = salinesurface;
  
  elem_slice = newreg;
  slicesurface = {lldomebase1+1,lldomebase2+1,lldomebase3+1,lldomebase4+1,llroids1+1,llroids2+1,llroids3+1,llroids4+1,llslicebase1+1,llslicebase2+1,llslicebase3+1,llslicebase4+1,llsliceds4+1,llsliceds1+1,llsliceds2+1,llsliceds3+1};
  Surface Loop(elem_slice) = {lldomebase1+1,lldomebase2+1,lldomebase3+1,lldomebase4+1,llroids1+1,llroids2+1,llroids3+1,llroids4+1,llslicebase1+1,llslicebase2+1,llslicebase3+1,llslicebase4+1,llsliceds4+1,llsliceds1+1,llsliceds2+1,llsliceds3+1};
  //Surface Loop(elem_slice) = slicesurface;

  elem_roi = newreg;
  roisurface = {llroidome1+1,llroidome2+1,llroidome3+1,llroidome4+1,llroids1+1,llroids2+1,llroids3+1,llroids4+1,llroislice2+1,llroislice1+1,llroislice4+1,llroislice3+1};
  Surface Loop(elem_roi) = {llroidome1+1,llroidome2+1,llroidome3+1,llroidome4+1,llroids1+1,llroids2+1,llroids3+1,llroids4+1,llroislice2+1,llroislice1+1,llroislice4+1,llroislice3+1};
  //Surface Loop(elem_roi) = roisurface;

Return

lcar3 = 1.5; lcar1 = 0.2;
x = 0.; y= 0.; z=0.; r=8.0; h=0.5; rs=1.;
Call SliceSaline3d ;

model_base = news;
Physical Surface (model_base) = elem_model_base;

model_dome = news;
Physical Surface (model_dome) = elem_dome;

salinesurf = news;
Physical Surface (salinesurf) = salinesurface;
salinevolume = newreg;
Volume(salinevolume) = elem_saline;
salinevol = newreg;
Physical Volume (salinevol) = salinevolume;

slicesurf = news;
Physical Surface (slicesurf) = slicesurface;
slicevolume = newreg;
Volume(slicevolume) = elem_slice;
slicevol = newreg;
Physical Volume (slicevol) = slicevolume;

roisurf = news;
Physical Surface (roisurf) = roisurface;
roivolume = newreg;
Volume(roivolume) = elem_roi;
roivol = newreg;
Physical Volume (roivol) = roivolume;
 