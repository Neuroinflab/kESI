DC=dolfin-convert $< $@
GMSH=gmsh -3 -optimize_netgen $<

all: eighth_of_sphere.xml slice.xml

eighth_of_sphere.msh: eighth_of_sphere.geo
	${GMSH}

eighth_of_sphere.xml: eighth_of_sphere.msh
	${DC}

slice.msh: slice.geo
	${GMSH}

slice.xml: slice.msh
	${DC}
