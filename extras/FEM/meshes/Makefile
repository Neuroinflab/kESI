DC=dolfin-convert $< $@
KC=python kesi_convert.py $< $@
GMSH=gmsh -3 -optimize_netgen $<
GMSH_NOOPT=gmsh -3 $<
SED=sed 's/$(1)/$(2)/g' $< > $@

EIGHTH_OF_SPHERE_XDMF=eighth_of_sphere.xdmf
EIGHTH_OF_SPHERE_XML=eighth_of_sphere.xml eighth_of_sphere_coarse.xml eighth_of_sphere_coarser.xml eighth_of_sphere_coarsest.xml
EIGHTH_OF_SPHERE_MESHES=$(EIGHTH_OF_SPHERE_XDMF) $(EIGHTH_OF_SPHERE_XML)
FINITE_SLICE_MESHES=finite_slice.xml finite_slice_small.xml finite_slice_smaller.xml finite_slice_small_coarse.xml finite_slice_smaller_coarse.xml
SPHERICAL_MESHES=one_sphere.msh two_spheres.xml four_spheres.xml four_spheres_composite.xml
MESHES=$(EIGHTH_OF_SPHERE_MESHES) $(FINITE_SLICE_MESHES) $(SPHERICAL_MESHES)

.SECONDARY: eighth_of_sphere.msh eighth_of_sphere_coarse.msh eighth_of_sphere_coarser.msh eighth_of_sphere_coarsest.msh finite_slice.msh finite_slice_small_coarse.msh finite_slice_smaller_coarse.msh one_sphere.msh two_spheres.msh four_spheres.msh four_spheres_composite.msh

all: $(MESHES)

%_coarsest.msh: %_coarsest.geo
	${GMSH_NOOPT}

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
