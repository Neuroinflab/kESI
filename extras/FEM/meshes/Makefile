all: eighth_of_sphere.xml

eighth_of_sphere.msh: eighth_of_sphere.geo
	gmsh -3 -optimize_netgen $<

eighth_of_sphere.xml: eighth_of_sphere.msh
	dolfin-convert $< $@

