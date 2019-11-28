Mesh.Algorithm = 5;

Function CheeseHole 
  // taken from t5.geo

  p1 = newp; Point(p1) = {x,  y,  z,  lcar1} ;
  p2 = newp; Point(p2) = {x+r,y,  z,  lcar3} ;
  p3 = newp; Point(p3) = {x,  y+r,z,  lcar3} ;
  p4 = newp; Point(p4) = {x,  y,  z+r,lcar3} ;
  p5 = newp; Point(p5) = {x-r,y,  z,  lcar3} ;
  p6 = newp; Point(p6) = {x,  y,  z-r,lcar3} ;
  p7 = newp; Point(p7) = {x,  y-h,  z,  lcar1} ;
  p8 = newp; Point(p8) = {x+r,y-h,  z,  lcar3} ;
  p9 = newp; Point(p9) = {x,  y-h,  z-r,  lcar3} ;
  p10 = newp; Point(p10) = {x,  y-h,  z+r,lcar3} ;
  p11 = newp; Point(p11) = {x-r,y-h,  z,  lcar3} ;
  p12 = newp; Point(p12) = {x+rs,y-h,  z,  lcar1} ;
  p13 = newp; Point(p13) = {x,  y-h,  z-rs,  lcar1} ;
  p14 = newp; Point(p14) = {x,  y-h,  z+rs,lcar1} ;
  p15 = newp; Point(p15) = {x-rs,y-h,  z,  lcar1} ;
  p16 = newp; Point(p16) = {x+rs,y,  z,  lcar1} ;
  p17 = newp; Point(p17) = {x,  y,  z-rs,  lcar1} ;
  p18 = newp; Point(p18) = {x,  y,  z+rs,lcar1} ;
  p19 = newp; Point(p19) = {x-rs,y,  z,  lcar1} ;

  c1 = newreg; Circle(c1) = {p16,p1,p18};
  c2 = newreg; Circle(c2) = {p18,p1,p19};
  c3 = newreg; Circle(c3) = {p19,p1,p17};
  c4 = newreg; Circle(c4) = {p17,p1,p16};
  c5 = newreg; Circle(c5) = {p2,p1,p3};
  c6 = newreg; Circle(c6) = {p3,p1,p5};
  c7 = newreg; Circle(c7) = {p6,p1,p3};
  c8 = newreg; Circle(c8) = {p3,p1,p4};
  c9 = newreg; Circle(c9) = {p10,p7,p8};
  c10 = newreg; Circle(c10) = {p9,p7,p8};
  c11 = newreg; Circle(c11) = {p9,p7,p11};
  c12 = newreg; Circle(c12) = {p11,p7,p10};
  c13 = newreg; Circle(c13) = {p2,p1,p6};
  c14 = newreg; Circle(c14) = {p6,p1,p5};
  c15 = newreg; Circle(c15) = {p5,p1,p4};
  c16 = newreg; Circle(c16) = {p4,p1,p2};
  c17 = newreg; Circle(c17) = {p12,p7,p14};
  c18 = newreg; Circle(c18) = {p14,p7,p15};
  c19 = newreg; Circle(c19) = {p15,p7,p13};
  c20 = newreg; Circle(c20) = {p13,p7,p12};
  
  line1 = newl; Line(line1) = {p1, p16};
  line2 = newl; Line(line2) = {p1, p17};
  line3 = newl; Line(line3) = {p1, p19};
  line4 = newl; Line(line4) = {p1, p18};
  line5 = newl; Line(line5) = {p2, p8};
  line6 = newl; Line(line6) = {p6, p9};
  line7 = newl; Line(line7) = {p5, p11};
  line8 = newl; Line(line8) = {p4, p10};
  line9 = newl; Line(line9) = {p12, p8};
  line10 = newl; Line(line10) = {p13, p9};
  line11 = newl; Line(line11) = {p15, p11};
  line12 = newl; Line(line12) = {p14, p10};
  line13 = newl; Line(line13) = {p12, p7};
  line14 = newl; Line(line14) = {p13, p7};
  line15 = newl; Line(line15) = {p15, p7};
  line16 = newl; Line(line16) = {p14, p7};
  line17 = newl; Line(line17) = {p2, p16};
  line18 = newl; Line(line18) = {p4, p18};
  line19 = newl; Line(line19) = {p5, p19};
  line20 = newl; Line(line20) = {p6, p17};
  line21 = newl; Line(line21) = {p12, p16};
  line22 = newl; Line(line22) = {p13, p17};
  line23 = newl; Line(line23) = {p15, p19};
  line24 = newl; Line(line24) = {p14, p18};

  l1 = newreg; Line Loop(l1) = {c5,c8,c16};   Ruled Surface(newreg) = {l1};
  l2 = newreg; Line Loop(l2) = {c7,-c5,c13};   Ruled Surface(newreg) = {l2};
  l3 = newreg; Line Loop(l3) = {-c8,c6,c15};  Ruled Surface(newreg) = {l3};
  l4 = newreg; Line Loop(l4) = {-c6,-c7,c14};  Ruled Surface(newreg) = {l4};
  l5 = newreg; Line Loop(l5) = {line1, c1, -line4}; Plane Surface(newreg) = {l5};
  l6 = newreg; Line Loop(l6) = {line4, c2, -line3}; Plane Surface(newreg) = {l6};
  l7 = newreg; Line Loop(l7) = {line3, c3, -line2}; Plane Surface(newreg) = {l7};
  l8 = newreg; Line Loop(l8) = {line2, c4, -line1}; Plane Surface(newreg) = {l8};
  l9 = newreg; Line Loop(l9) = {line17, c16, -line18,c1};   Ruled Surface(newreg) = {l9};
  l10 = newreg; Line Loop(l10) = {c2, line18, c15, -line19};   Ruled Surface(newreg) = {l10};
  l11 = newreg; Line Loop(l11) = {c3, line19, c14, -line20};  Ruled Surface(newreg) = {l11};
  l12 = newreg; Line Loop(l12) = {c4, line20, c13, -line17};  Ruled Surface(newreg) = {l12};
  l13 = newreg; Line Loop(l13) = {c1, line21, -c17, -line24};   Ruled Surface(newreg) = {l13};
  l14 = newreg; Line Loop(l14) = {c2, line24, -c18, -line23};   Ruled Surface(newreg) = {l14};
  l15 = newreg; Line Loop(l15) = {c3, line23, -c19, -line22};  Ruled Surface(newreg) = {l15};
  l16 = newreg; Line Loop(l16) = {c4, line22, -c20, -line21};  Ruled Surface(newreg) = {l16};
  l17 = newreg; Line Loop(l17) = {line15, c18, -line16}; Ruled Surface(newreg) = {l17};
  l18 = newreg; Line Loop(l18) = {line16, c17, -line13}; Ruled Surface(newreg) = {l18};
  l19 = newreg; Line Loop(l19) = {line13, c20, -line14}; Ruled Surface(newreg) = {l19};
  l20 = newreg; Line Loop(l20) = {line14, c19, -line15}; Ruled Surface(newreg) = {l20};
  l21 = newreg; Line Loop(l21) = {c17, -line9, c9, line12};   Ruled Surface(newreg) = {l21};
  l22 = newreg; Line Loop(l22) = {c18, -line12, c12, line11};   Ruled Surface(newreg) = {l22};
  l23 = newreg; Line Loop(l23) = {c19, -line11, c11, line10};  Ruled Surface(newreg) = {l23};
  l24 = newreg; Line Loop(l24) = {c20, -line10, -c10, line9};  Ruled Surface(newreg) = {l24};
  l25 = newreg; Line Loop(l25) = {c13, line6, c10, -line5};   Ruled Surface(newreg) = {l25};
  l26 = newreg; Line Loop(l26) = {c16, line5, -c9, -line8};   Ruled Surface(newreg) = {l26};
  l27 = newreg; Line Loop(l27) = {c15, line8, -c12, -line7};  Ruled Surface(newreg) = {l27};
  l28 = newreg; Line Loop(l28) = {c14, line7, -c11, -line6};  Ruled Surface(newreg) = {l28};
  //l29 = newreg; Line Loop(l29) = {line24, -line18, line8, line12};   Plane Surface(newreg) = {l29};
  //l30 = newreg; Line Loop(l30) = {c16, line5, -c9, -line8};   Ruled Surface(newreg) = {l30};
  //l31 = newreg; Line Loop(l31) = {c15, line8, -c12, -line7};  Ruled Surface(newreg) = {l31};
  //l32 = newreg; Line Loop(l32) = {c14, line7, -c11, -line6};  Ruled Surface(newreg) = {l32};

  elem_dome = newreg;
  Surface Loop(elem_dome) = {l4+1,l3+1,l1+1,l2+1,l25+1,l26+1,l27+1,l28+1};

  elem_model_base = newreg;
  Surface Loop(elem_model_base) = {l17+1,l18+1,l19+1,l20+1,l21+1,l22+1,l23+1,l24+1};

  elem_saline = newreg;
  Surface Loop(elem_saline) = {l1+1,l2+1,l3+1,l4+1,l5+1,l6+1,l7+1,l8+1,l9+1,l10+1,l11+1,l12+1};
  
  elem_slice = newreg;
  Surface Loop(elem_slice) = {l9+1,l10+1,l11+1,l12+1,l13+1,l14+1,l15+1,l16+1,l21+1,l22+1,l23+1,l24+1,l25+1,l26+1,l27+1,l28+1};

  elem_roi = newreg;
  Surface Loop(elem_roi) = {l5+1,l6+1,l7+1,l8+1,l13+1,l14+1,l15+1,l16+1,l17+1,l18+1,l19+1,l20+1};

Return

lcar3 = 1.5; lcar1 = 0.2;
x = 0.; y= 0.; z=0.; r=8.0; t=1; h=3.; rs=1.5;
Call CheeseHole ;
model_base = news;
Physical Surface (model_base) = {l17+1,l18+1,l19+1,l20+1,l21+1,l22+1,l23+1,l24+1};

model_dome = news;
Physical Surface (model_dome) = {l4+1,l3+1,l1+1,l2+1,l25+1,l26+1,l27+1,l28+1};

salinesurf = news;
Physical Surface (salinesurf) = {l1+1,l2+1,l3+1,l4+1,l5+1,l6+1,l7+1,l8+1,l9+1,l10+1,l11+1,l12+1};
salinevolume = newreg;
Volume(salinevolume) = elem_saline;
salinevol = newreg;
Physical Volume (salinevol) = salinevolume;

slicesurf = news;
Physical Surface (slicesurf) = {l9+1,l10+1,l11+1,l12+1,l13+1,l14+1,l15+1,l16+1,l21+1,l22+1,l23+1,l24+1,l25+1,l26+1,l27+1,l28+1};
slicevolume = newreg;
Volume(slicevolume) = elem_slice;
slicevol = newreg;
Physical Volume (slicevol) = slicevolume;

roisurf = news;
Physical Surface (roisurf) = {l5+1,l6+1,l7+1,l8+1,l13+1,l14+1,l15+1,l16+1,l17+1,l18+1,l19+1,l20+1};
roivolume = newreg;
Volume(roivolume) = elem_roi;
roivol = newreg;
Physical Volume (roivol) = roivolume;
 