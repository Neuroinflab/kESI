DC=dolfin-convert $< $@
KC=python kesi_convert.py $< $@
GMSH=gmsh -3 -optimize_netgen $<

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
