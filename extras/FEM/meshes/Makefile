DC=dolfin-convert $< $@
KC=python kesi_convert.py $< $@
GMSH=gmsh -3 -optimize_netgen $<
SED=sed 's/$(1)/$(2)/g' $< > $@

all: eighth_of_sphere.xml finite_slice.xml

all: $(MESHES)

%.msh: %.geo
	${GMSH}

finite_slice.xml: finite_slice.msh
	${DC}

%.xml %_physical_region.xml %_facet_region.xml: %.msh
	${DC}

%.xdmf %_boundaries.xdmf %_subdomains.xdmf: %.msh
	${KC}

eighth_of_sphere.geo: eighth_of_sphere.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}

eighth_of_sphere_coarse.geo: eighth_of_sphere.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,2.0}

eighth_of_sphere_coarser.geo: eighth_of_sphere.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,4.0}

eighth_of_sphere_coarsest.geo: eighth_of_sphere.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,8.0}
