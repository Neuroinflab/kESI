DC=dolfin-convert $< $@
KC=python kesi_convert.py $< $@
GMSH=gmsh -3 -optimize_netgen $<
GMSH_NOOPT=gmsh -3 $<
SED=sed 's/$(1)/$(2)/g' $< > $@
SED_INPLACE=sed -i 's/$(1)/$(2)/g' $@

EIGHTH_OF_SPHERE_XDMF=eighth_of_sphere.xdmf
EIGHTH_OF_SPHERE_XML=eighth_of_sphere.xml eighth_of_sphere_coarse.xml eighth_of_sphere_coarser.xml eighth_of_sphere_coarsest.xml
EIGHTH_OF_SPHERE_MESHES=$(EIGHTH_OF_SPHERE_XDMF) $(EIGHTH_OF_SPHERE_XML)
FINITE_SLICE_MESHES=finite_slice.xml finite_slice_small.xml finite_slice_smaller.xml finite_slice_small_coarse.xml finite_slice_smaller_coarse.xml
SPHERICAL_MESHES=one_sphere.xml two_spheres.xml four_spheres.xml four_spheres_composite.xml
WEDGE_MESHES=eighth_wedge_of_one_sphere.xml eighth_wedge_of_two_spheres.xml eighth_wedge_of_four_spheres.xml eighth_wedge_of_four_spheres_composite.xml
WEDGE_MESHES_ALL=$(WEDGE_MESHES) $(WEDGE_MESHES:.xml=_fine.xml) $(WEDGE_MESHES:.xml=_finer.xml) $(WEDGE_MESHES:.xml=_finest.xml) $(WEDGE_MESHES:.xml=_coarse.xml) $(WEDGE_MESHES:.xml=_coarser.xml) $(WEDGE_MESHES:.xml=_coarsest.xml)
MESHES=$(EIGHTH_OF_SPHERE_MESHES) $(FINITE_SLICE_MESHES) $(SPHERICAL_MESHES) $(WEDGE_MESHES_ALL)

MESHES_MSH=$(MESHES:.xml=.msh)

.SECONDARY: $(MESHES_MSH) $(WEDGE_MESHES_ALL:.xml=.geo)

.PHONY: all

all: $(MESHES)

eighth_wedge_of_one_sphere_coarse.msh eighth_wedge_of_two_spheres_coarse.msh eighth_wedge_of_two_spheres_coarser.msh eighth_wedge_of_two_spheres_coarsest.msh eighth_wedge_of_four_spheres_coarse.msh eighth_wedge_of_four_spheres_coarser.msh eighth_wedge_of_four_spheres_coarsest.msh: %.msh: %.geo
	${GMSH_NOOPT}

# TODO: validate whether disabling of optimization is necessary
eighth_wedge_of_two_spheres.msh eighth_wedge_of_four_spheres.msh eighth_wedge_of_one_sphere_fine.msh eighth_wedge_of_one_sphere_finer.msh eighth_wedge_of_one_sphere_finest.msh eighth_wedge_of_two_spheres_fine.msh  eighth_wedge_of_two_spheres_finer.msh eighth_wedge_of_four_spheres_finer.msh eighth_wedge_of_four_spheres_finest.msh eighth_wedge_of_four_spheres_composite_fine.msh eighth_wedge_of_four_spheres_composite_finer.msh eighth_wedge_of_four_spheres_composite_finest.msh: %.msh: %.geo
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


# EIGHTH WEDGE SPHERICAL GEOMETRIES

eighth_wedge_of_%_fine.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,0.5}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}

eighth_wedge_of_%_finer.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,0.25}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}

eighth_wedge_of_%_finest.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,0.125}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}

eighth_wedge_of_%_coarse.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,2.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}

eighth_wedge_of_%_coarser.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,4.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}

eighth_wedge_of_%_coarsest.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,4.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}

eighth_wedge_of_%.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,8}


# SIXTH WEDGE SPHERICAL GEOMETRIES

sixth_wedge_of_%_fine.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,0.5}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}

sixth_wedge_of_%_finer.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,0.25}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}

sixth_wedge_of_%_finest.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,0.125}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}

sixth_wedge_of_%_coarse.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,2.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}

sixth_wedge_of_%_coarser.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,4.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}

sixth_wedge_of_%_coarsest.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,4.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}

sixth_wedge_of_%.geo: wedge_of_%.geo.template
	${call SED,SED_RELATIVE_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_RELATIVE_ROI_ELEMENT_SIZE,1.0}
	${call SED_INPLACE,SED_NUMBER_OF_WEDGES,6}
