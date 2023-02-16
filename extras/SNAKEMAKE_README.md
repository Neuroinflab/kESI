# Snakefile Documentation

## Paths

Unless stated otherwise, all paths in this section are relative to the readme
directory, that is the _extras/_ directory of the working copy.


### Meshes

All meshes are stored in the _FEM/meshes_ directory containing a _Snakefile_
with mesh compilation workflows, _gmsh_ geometry files (_*.geo_) and their
templates (_*.geo.template_).

Most of compiled meshes are stored in the _FEM/meshes/meshes_ directory
which does not contain any tracked files, thus it may be softlinked or mounted
as a dedicated filesystem.  Paths to files in the directory follow pattern
_FEM/meshes/meshes/\<geometry\>/\<mesh granularity\>\<suffix\>_
where:
- _\<geometry\>_ is the stem of the geometry template file
(_FEM/meshes/\<stem\>.geo.template_),
- _\<mesh granularity\>_ is either of _coarsest_, _coarser_, _coarse_, _normal_,
  _fine_, _finer_, _finest_, _superfine_, _superfinest_ (in descending order
  of granularity),
- _suffix_ determines content of the file(files).

| suffix                                    | content               |
|-------------------------------------------|-----------------------|
| _.geo_                                    | _gmsh_ geometry       |
| _.msh_                                    | _gmsh_ mesh           |
| _.xdmf_ and _.h5_                         | _FEniCS_ mesh         |
| _\_subdomains.xdmf_ and _\_subdomains.h5_ | _FEniCS_ mesh domains |
| _\_boundaries.xdmf_ and _\_boundaries.h5_ | _FEniCS_ boundaries   |


### Solutions

Most of kESI data is stored in the _FEM/solutions_ directory.
It does not  contain any tracked files, thus it may be softlinked
or mounted as a dedicated filesystem.

The _FEM/solutions/tutorial_ directory contains tutorial-related data,
thus its content is discussed therein.

The _FEM/solutions/paper_ directory contains publication-related data.
For the sake of readability the **_FEM/solutions/paper_ path is omitted
in the further text**, that is all paths are relative to the root
of the _FEM/solutions/paper_ subtree unless stated otherwise.
