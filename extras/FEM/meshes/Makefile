DC=dolfin-convert $< $@
KC=python kesi_convert.py $< $@
GMSH=gmsh -3 -optimize_netgen $<
SED=sed 's/$(1)/$(2)/g' $< > $@

EIGHTH_OF_SPHERE_XDMF=eighth_of_sphere.xdmf
EIGHTH_OF_SPHERE_XML=eighth_of_sphere.xml eighth_of_sphere_coarse.xml eighth_of_sphere_coarser.xml eighth_of_sphere_coarsest.xml
EIGHTH_OF_SPHERE_MESHES=$(EIGHTH_OF_SPHERE_XDMF) $(EIGHTH_OF_SPHERE_XML)
MESHES=$(EIGHTH_OF_SPHERE_MESHES) finite_slice.xml

all: $(MESHES)

%.msh: %.geo
	${GMSH}


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
