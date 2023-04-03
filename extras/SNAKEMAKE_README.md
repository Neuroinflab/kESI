# Snakefile Documentation

Unless stated otherwise, all paths in this document are relative to the readme
directory, that is the `extras/` directory of the working copy.


## Data

Unless stated otherwise, all paths in this section are relative to the `data/`
directory, that is the `extras/data` directory of the working copy.

### Bundled

<!--Unless stated otherwise, all paths in this section are relative to the `bundled/`
directory, that is the `extras/data/bundled/` directory of the working copy.-->

kESI comes with exemplary source data which are not the part of the method itself.  
The subtree contains four directories:
```
bundled/
  csd_basis_functions/
  electrode_locations/
  meshes/
  model_properties/  
```
where:

| subdirectory          | described in                                                                                                                  |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `csd_basis_functions` | [CSD basis functions](#data-bundled-csd_basis_functions)                                                                                                      |
| `electrode_locations` | [Position of electrodes](#data-bundled-position_of_electrodes)                                                                |
| `meshes`              | [Mesh geometry files](#data-bundled-mesh_geometry_files) and [Mesh geometry templates](#data-bundled-mesh_geometry_templates) |
| `model_properties`    | [Model properties](#data-bundled-model_properties)                                                                            |

If you want to provide Snakefile with custom source files, the subtree is a proper place
to store them.  

#### CSD basis functions <a name="data-bundled-csd_basis_functions"></a>

Shape definitions of CSD profiles (including size)
for `SphericalSplineSourceBase` class (and its descendant
`SphericalSplineSourceKCSD`).  The profiles are centered 
at `(0, 0, 0)`, which make them so-called _model sources_.


#### Position of electrodes <a name="data-bundled-position_of_electrodes"></a>

Locations of electrodes are stored in `*.ini` files, 
where section names are names of the respective point electrodes 
and fields of a section (_'x'_, _'y'_, _'z'_) are coordinates of the electrode.

In case you want to provide your own files mind that double underscores (`__`)
are allowed neither in names of files nor directories in that directory subtree.


#### Mesh geometry files <a name="data-bundled-mesh_geometry_files"></a>

At the moment there is only one bundled file (`circular_slice_composite.geo`)
in the _gmsh_ geometry format.


#### Mesh geometry templates <a name="data-bundled-mesh_geometry_templates"></a>

Filenames of templates of _gmsh_ geometry files follow the `<stem>.geo.template`
pattern, where `<stem>` determines most mesh properties but
size of its particular elements (which is controlled by the
`SED_RELATIVE_ELEMENT_SIZE` template marker).


#### Model properties <a name="data-bundled-model_properties"></a>

Physical properties of the model (i.e. medium conductivity) and
its geometrical properties (e.g. sphere radius) are stored in the
`*ini` files.


### Generated

Unless stated otherwise, all paths in this section are relative to the `generated/`
directory, that is the `extras/data/generated/` directory of the working copy.
As no bundled files are provided therein, the directory subtree may
be not only a regular directory tree, but also either a mounted filesystem
or a soft link.

The subtree looks as follows:
```
generated/
  meshes/
  setups/
  fenics_leadfield_corrections/
  sampled_leadfield_corrections
  potential_basis_functions/
  kernel/
  csd_profiles/
```

where

| subdirectory                    | is described in                                                                |
|---------------------------------|--------------------------------------------------------------------------------|
| `meshes`                        | [Meshes](#data-generated-meshes)                                               |
| `setups`                        | [Setups](#data-generated-setups)                                               |
| `fenics_leadfield_corrections`  | [Fenics leadfield corrections](#data-generated-fenics_leadfield_corrections)   |
| `sampled_leadfield_corrections` | [Sampled leadfield corrections](#data-generated-sampled_leadfield_corrections) |
| `potential_basis_functions`     | [Potential basis functions](#data-generated-potential_basis_functions)         |
| `kernel`                        | [Kernel](#data-generated-kernel)                                               |
| `csd_profiles`                  | [CSD profiles](#data-generated_csd_profiles)                                   |


#### Meshes <a name="data-generated-meshes"></a>

All meshes are stored in the `meshes/` directory.  The filesystem subtree
follows the pattern:
```
meshes/
  <geometry>__<version>/
    <granularity>.geo
    <granularity>.msh
    <granularity>.xdmf
    <granularity>.h5
    <granularity>_subdomains.xdmf
    <granularity>_subdomains.h5
    <granularity>_boundaries.xdmf
    <granularity>_boundaries.h5
```
where:
- `<geometry>` is the stem of the
  [mesh geometry template file](#data-bundled-mesh_geometry_templates),
- `<granularity>` determines mesh granularity (size of mesh elements) 
  if mesh is derived from a geometry template.  Relative element sizes
  are listed in the table below.  Other values of `<granularity>` are
  also allowed (e.g. for meshes not derived from templates).

| `<granularity>` | `SED_RELATIVE_ELEMENT_SIZE` |
|-----------------|-----------------------------|
| `coarsest`      | 8.0                         |
| `coarser`       | 4.0                         |
| `coarse`        | 2.0                         |
| `normal`        | 1.0                         |
| `fine`          | 0.5                         |
| `finer`         | 0.25                        |
| `fines`         | 0.125                       |
| `superfine`     | 0.0625                      |
| `superfinest`   | 0.03125                     |

Content of a file depends on suffix of the filename:

| filename                                  | content               |
|-------------------------------------------|-----------------------|
| `*.geo`                                   | _gmsh_ geometry       |
| `*.msh`                                   | _gmsh_ mesh           |
| `*.xdmf` and `*.h5`                       | _FEniCS_ mesh         |
| `*_subdomains.xdmf` and `*_subdomains.h5` | _FEniCS_ mesh domains |
| `*_boundaries.xdmf` and `*_boundaries.h5` | _FEniCS_ boundaries   |


#### Setups <a name="data-generated-setups"></a>

Setups are stored in `setups/` directory.  The `*.csv` files contain a table
with location of the electrodes, with columns named intuitively:
_NAME_ (name of the electrode), _X_, _Y_ and _Z_ (its location coordinates
in meters).  An examplary file may look like:
```
NAME,X,Y,Z
A_00,-0.006,0.0,0.046
A_01,-0.006,0.0,0.0485
```

Filenames follow the `[<aaa>__[<bbb>__[...]]]<zzz>.csv` pattern
(part in `[]` is optional and `...` is used instead of further recursion),
encoding path to [the bundled file](#data-bundled-position_of_electrodes)
the locations of electrodes were copied from, that is to:
```
<root of the working directory>/
  extras/
    data/
      bundled/
        electrode_positions/
          [aaa/
            [bbb/
              [...]]]
                zzz.ini
```


#### Fenics leadfield corrections <a name="data-generated-fenics_leadfield_corrections"></a>

`fenics_leadfield_corrections/`


#### Sampled leadfield corrections <a name="data-generated-sampled_leadfield_corrections"></a>

`sampled_leadfield_corrections/`


#### Potential basis functions <a name="data-generated-potential_basis_functions"></a>

Transition from basis function in the CSD space to basis function in potentials space.

`potential_basis_functions/`


#### Kernels <a name="data-generated-kernels"></a>

`kernels/`


#### CSD profiles <a name="data-generated-csd_profiles"></a>

```
csd_profiles/
  <setup>/
    <subsetup>/
      <csd_basis_functions>/
        <path>==kCSD/<conductivity [S/m]>/ |
                kESI/<sampling>/<model>/<mesh path>/<degree>/ |
                mixed/<conductivity [S/m]>/<sampling>/<model>/<mesh path>/<degree>/ - path
                different for kCSD, kESI and mixed source models
          <csd_grid>
            <sources>.npz==eigensources.npz
            <fwd path>==<fwd model>/<fwd mesh path>/<fwd degree>/
              <sources>.csv
```
where:

| wildcard                | meaning                                                                                                                                          |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `<setup>`               | set of possible electrodes locations for a given coordinate system                                                                               |
| `<subsetup>`            | subset of electrodes locations, useful for optimization purposes                                                                                 |
| `<csd_basis_functions>` | defines component part of cross kernel, number of basis, their location in the space and the shape of CSD profiles                               |
| `<path>`                | path is different for kCSD, kESI and mixed (mixed in 1:1 ratio kCSD and kESI eigensources for the same basis function in CSD space) source model |
| `<sampling>`            | regular, rectangular grid on which kESI leadfield correction was obtained                                                                        |
| `<model>`               | defines geometrical and physical properties of the forward model                                                                                 |
| `<mesh path>`           | defines mesh in the space of the model, depends on geometrical properties                                                                        |
| `<degree>`              | informs elements of what degree are span on the grid, it's independent of the model                                                              |
| `<csd_grid>`            |                                                                                                                                                  |
| `<sources>.npz`         |                                                                                                                                                  |
| `<fwd path>`            |                                                                                                                                                  |
| `<fwd model>`           |                                                                                                                                                  |
| `<fwd mesh path>`       |                                                                                                                                                  |
| `<fwd degree>`          |                                                                                                                                                  |
| `<sources>.csv`         |                                                                                                                                                  |


`<model>/<mesh path>/<degree>/` - completely defines FEM model




# OLD BELOW

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

Directories following pattern _\<geometry\>/\<granularity\>/\<degree\>/_
contain corrections of leadfields of electrodes, which are crucial for kESI.
File _\<geometry\>/\<granularity\>/\<degree\>/\<electrode\>.h5_ contains
correction of the leadfield of the electrode _\<electrode\>_ saved as a _FEniCS_
function and file _\<geometry\>/\<granularity\>/\<degree\>/\<electrode\>.ini_
contains its metadata.
File _\<geometry\>/\<granularity\>/\<degree\>/sampled/\<k\>/\<electrode\>.npz_
contains the **sampled leadfield correction** of the electrode saved in _NumPy_
format.  The correction is sampled on a regular, $(2^k + 1)^3$ grid.

Directories following pattern
_images/\<geometry\>/\<granularity\>/\<degree\>/kernels/\<k\>/_
contain kESI/kCSD kernel-related files derived from
_\<geometry\>/\<granularity\>/\<degree\>/sampled/\<k\>/*.npz_
sampled leadfields.

| file                           | content                                                        |
|--------------------------------|----------------------------------------------------------------|
| _electrodes.csv_               | positions of electrodes                                        |
| _src_mask.npz_                 | positions of source centroids                                  |
| _\<method\>\_phi.npz_          | the transfer matrix ($\Phi$) of _\<method\>_                   |
| _\<method\>\_kernel.npz_       | the kernel matrix of _\<method\>_                              |
| _\<method\>\_crosskernel.npz_  | the volumetric crosskernel tensor of _\<method\>_              |
| _\<method\>\_analysis.npz_     | auxilary analytical data of _\<method\>_                       |
| _\<method\>\_eigensources.npz_ | the volumetric eigensource tensor of _\<method\>_              |
| _fair\_sources.npz_            | average of appropriate volumetric eigensources of both methods |

_\<method\>_ may be either _kCSD_ or _kESI_.  Note that
_images/\<geometry\>/\<granularity\>/\<degree\>/kernels/\<k\>/kCSD\_*.npz_
files are redundant.

In the _images/\<geometry\>/\<granularity\>/\<degree\>/kernels/\<k\>/**images**_
subtree (_\<inverse model\>_) results of the forward modelling are stored.  File
_\<inverse model\>/\<geometry\>/\<granularity\>/\<degree\>/**\<sources\>**.csv_
contains potentials at the electrodes generated by CSD profiles from
_\<inverse model\>/../**\<sources\>**.npz_ with a FEM forward model appropriate
for the _\<geometry\>/\<granularity\>/\<degree\>_ subpath.


## Files

### Leadfield correction function

A _FEniCS_ 3D scalar function \[ $V/A$ \].

### Leadfield correction metadata

An _*.ini_ file.

| section    | field                     | value                                                    |
|------------|---------------------------|----------------------------------------------------------|
| fem        | mesh                      | path to the main mesh file                               |
|            | degree                    | degree of the element                                    |
|            | element_type              | type of the element                                      |
| model      | config                    | path to model properties (conductivity etc.)             |
| electrode  | x                         | X coordinate of the point electrode \[ $m$ \]            |
|            | y                         | Y coordinate of the point electrode \[ $m$ \]            |
|            | z                         | Z coordinate of the point electrode \[ $m$ \]            |
| correction | global_preprocessing_time | location-independent preprocessing time \[ $s$ \]        |
|            | setup_time                | total function manager and FEM initiation time \[ $s$ \] |
|            | total_solving_time        | total time of location-dependent processing \[ $s$ \]    |
|            | local_preprocessing_time  | location-dependent preprocessing time \[ $s$ \]          |
|            | solving_time              | time of FEM equation solving \[ $s$ \]                   |
|            | base_conductivity         | base conductivity used by renormalization \[ $S/m$ \]    |
|            | filename                  | relative path to the correction function                 |


### Sampled leadfield correction

A compressed NumPy file (_*.npz_).

| array                   | shape                                         | type              | content                                           |
|-------------------------|-----------------------------------------------|-------------------|---------------------------------------------------|
| _CORRECTION\_POTENTIAL_ | $n^{POT}_x \times n^{POT}_y \times n^{POT}_z$ | float \[ $V/A$ \] | sampled leadfield correction                      |
| _X_                     | $n^{POT}_x$                                   | float \[ $m$ \]   | X nodes of the sampling grid                      |
| _Y_                     | $n^{POT}_y$                                   | float \[ $m$ \]   | Y nodes of the sampling grid                      |
| _Z_                     | $n^{POT}_z$                                   | float \[ $m$ \]   | Z nodes of the sampling grid                      |
| _LOCATION_              | $3$                                           | float \[ $m$ \]   | X, Y, Z coordinates of the electrode              |
| _BASE\_CONDUCTIVITY_    | scalar                                        | float \[ $S/m$ \] | base conductivity used by renormalization         |
| _\_PREPROCESSING\_TIME_ | scalar                                        | float \[ $s$ \]   | construction time of the `FunctionManager` object |
| _\_LOADING\_TIME_       | scalar                                        | float \[ $s$ \]   | loading time of the leadfield correction function |
| _\_PROCESSING\_TIME_    | scalar                                        | float \[ $s$ \]   | leadfield correction sampling time                |


### Position of electrodes

A CSV file.

| field  | type            | content                     |
|--------|-----------------|-----------------------------|
| _NAME_ | str             | name of the electrode       |
| _X_    | float \[ $m$ \] | X position of the electrode |
| _Y_    | float \[ $m$ \] | Y position of the electrode |
| _Z_    | float \[ $m$ \] | Z position of the electrode |


### Positions of source centroids

A compressed NumPy file (_*.npz_).

| array  | shape                                         | type            | content                      |
|--------|-----------------------------------------------|-----------------|------------------------------|
| _MASK_ | $n^{SRC}_x \times n^{SRC}_y \times n^{SRC}_z$ | bool            | mask of nodes with centroids |
| _X_    | $n^{SRC}_x \times 1 \times 1$                 | float \[ $m$ \] | X nodes of the centroid grid |
| _Y_    | $1 \times n^{SRC}_y \times 1$                 | float \[ $m$ \] | Y nodes of the centroid grid |
| _Z_    | $1 \times 1 \times n^{SRC}_z$                 | float \[ $m$ \] | Z nodes of the centroid grid |

`MASK.sum() == m` where `m` is the number of base functions.


### Transfer matrix

($\Phi$ matrix)

A compressed NumPy file (_*.npz_).

| array | shape        | type            | content                                                         |
|-------|--------------|-----------------|-----------------------------------------------------------------|
| _PHI_ | $m \times n$ | float \[ $V$ \] | `PHI[i, j]` is value of `i`th base function at `j`-th electrode |


### Kernel matrix

A compressed NumPy file (_*.npz_).

| array    | shape        | type              | content               |
|----------|--------------|-------------------|-----------------------|
| _KERNEL_ | $n \times n$ | float \[ $V^2$ \] | the kernel matrix $K$ |

$$
K = \Phi^T \Phi
$$


### Auxilary analytical data

A compressed NumPy file (_*.npz_).

| array          | shape        | type            | content                                   |
|----------------|--------------|-----------------|-------------------------------------------|
| _EIGENVALUES_  | $n$          | float \[ $V$ \] | Kernel eigenvalues ($\lambda = \Sigma^2$) |
| _EIGENSOURCES_ | $m \times n$ | float           | Eigensources in the cananical form $U$    |
| _LAMBDAS_      | $n$          | float           | $\Phi$ singular values ($\Sigma$)         |
| _EIGENVECTORS_ | $n \times n$ | float           | Kernel eigenvalues $V$                    |

$$
\Phi = U \Sigma V^T
$$

$$
K = V \lambda V^T
$$


### Volumetric cross-kernel tensor

A compressed NumPy file (_*.npz_).

| array         | shape                                                  | type                  | content                                 |
|---------------|--------------------------------------------------------|-----------------------|-----------------------------------------|
| _CROSSKERNEL_ | $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times n$ | float \[ $W / m^3$ \] | The crosskernel tensor ($\overline{K}$) |
| _X_           | $n^{CSD}_x \times 1 \times 1$                          | float \[ $m$ \]       | X nodes of the CSD sampling grid        |
| _Y_           | $1 \times n^{CSD}_y \times 1$                          | float \[ $m$ \]       | Y nodes of the CSD sampling grid        |
| _Z_           | $1 \times 1 \times n^{CSD}_z$                          | float \[ $m$ \]       | Z nodes of the CSD sampling grid        |

The tensor yields volumetric CSD reconstruction $C = \overline{K} K^{-1} V$,
where $V$ is a vector (matrix in case of timepoints) of measured potentials and
$C$ is an $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z$ array.


### Volumetric eigensource tensor

A compressed NumPy file (_*.npz_).

| array   | shape                                                  | type                  | content                          |
|---------|--------------------------------------------------------|-----------------------|----------------------------------|
| _CSD_   | $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times n$ | float \[ $A / m^3$ \] | sampled CSDs of n eigensources   |
| _X_     | $n^{CSD}_x \times 1 \times 1$                          | float \[ $m$ \]       | X nodes of the CSD sampling grid |
| _Y_     | $1 \times n^{CSD}_y \times 1$                          | float \[ $m$ \]       | Y nodes of the CSD sampling grid |
| _Z_     | $1 \times 1 \times n^{CSD}_z$                          | float \[ $m$ \]       | Z nodes of the CSD sampling grid |


### Potentials at electrodes generated by CSD profiles

A CSV file.

| field           | type            | content                                                   |
|-----------------|-----------------|-----------------------------------------------------------|
| _NAME_          | str             | name of the electrode                                     |
| _X_             | float \[ $m$ \] | X position of the electrode                               |
| _Y_             | float \[ $m$ \] | Y position of the electrode                               |
| _Z_             | float \[ $m$ \] | Z position of the electrode                               |
| _SOURCE\_\<i\>_ | float \[ $V$ \] | potential generated by the _i_-th source at the electrode |


## Tools

### Leadfield correction solving

_paper\_solve\_slice\_on\_plate.py_ and _paper\_solve\_sphere\_on\_plate.py_
take mesh, electrode location and model properties as input and calculate
the leadfield correction _FEniCS_ function.

| argument         | description                                                                                              |
|------------------|----------------------------------------------------------------------------------------------------------|
| `--output`       | path to the metadata file (function filename is inferred by substituting the '.ini' extension with '.h5' |
| `--config`       | path to the model config file                                                                            |
| `--electrodes`   | path to the electrode location file                                                                      |
| `--name`         | name of the electrode                                                                                    |
| `--mesh`         | path to the main _FEniCS_ mesh                                                                           |
| `--degree`       | degree of the FEM element (defaults to 1)                                                                |
| `--element-type` | type of the FEM element (defaults to `'CG'` (Continuous Galerkin)                                        |
| `--quiet`        | supress control messages                                                                                 |

#### Slice specific

The `--ground-potential` parameter is the potantial \[ $V$ \] at the grounded
slice-covering dome.  If not given a $0 V$ grounding is assumed at infinity.


#### Sphere specific

The `--grounded-plate-edge-z` parameter is the Z coordinate \[ $m$ \] of the
edge of the grounded ($0 V$ potential) conductive plate.  Defaults to $-88 mm$.


### Leadfield correction sampling

_paper\_sample\_slice\_solution.py_ and _paper\_sample\_spherical\_solution.py_
take _FEniCS_ function and sample it on a regular grid.

Forces:
- shape of _POT_ sampling grid to $2^k + 1 \times 2^k + 1 \times 2^k + 1$,
- shape and location of the sampling area,
- sampling nodes subset.

| argument   | description                                                          |
|------------|----------------------------------------------------------------------|
| `--output` | path to the output _*.npz_ file                                      |
| `--config` | path to the leadfield correction metadata                            |
| `-k`       | the $k$ parameter of the sampling grid (defaults to 9)               |
| `--fill`   | fill value for not sampled points of the grid (defaults to `np.nan`) |
| `--quiet`  | supress control messages                                             |

#### Slice specific

The `--sampling-radius` parameter is the edge length \[ $m$ \] of the sampled
cube.  It defaults to $0.3 mm$.

The cube is based at the $Z = 0$ plane and centered at $X = Y = 0$ axis.


#### Sphere specific

The `--sampling-radius` parameter is the radius \[ $m$ \] of the sampled sphere
centered at the beginning of the coordinate system.  It defaults to $79 mm$.


### Kernel calculation

_paper\_calculate\_kernels\_slice.py_
and _paper\_calculate\_kernels\_four\_spheres.py_ from sampled leadfield
corrections calculate kernels (and related matrices) for both kESI and kCSD
(redundant - the same for different kESI configurations).

Contains information about:
- _POT_, _CSD_, _SRC_ grid (subset of the sampling grid),
- source size,
- source profile,
- base function distribution.

| argument     | description                                              |
|--------------|----------------------------------------------------------|
| `electrodes` | names of electrodes to be used                           |
| `--input`    | path to the directory with sampled leadfield corrections |
| `--output`   | path to the directory for output files                   |
| `-k`         | the $k$ parameter of the Romberg integration method      |

The $k$ parameter limits radius of the CSD base function to $2^{k - 1} dx$,
where $dx$ is spacing of the X axis grid.  It is implicitly assumed that spacing
of all axes is equal (or at least that grids of Y and Z axes have no smaller
spacing than grid of the X axis).

#### Slice specific

Limits source centroids to $|y| <= h / 4$,
where $h = 0.3 mm$ is the slice thickness,
and adjusts grids appropriately.


#### Four spheres specific

Limits source centroids to $|x|, |y| < 2 cm$, $2.5 cm \leq z$ and
$r \leq 7.9 cm - 2^{k-1} dx$, and adjusts grids appropriately.


### Eigensource mixing

_paper\_mix\_eigensources.py_ loads two sets of volumetric CSD eigensources,
matches them (by maximizing the absolute value of the dot product of their
canonical rapresentation) and yields averaged volumetric sources.

Note that both sets must share at least shape of the transfer matrix ($\Phi$).

| argument   | description                            |
|------------|----------------------------------------|
| `input`    | two prefixes of the input eigensources |
| `--output` | path to the output _*.npz_ file        |


### Forward modelling

_paper\_forward\_model\_slice.py_ and _paper\_forward\_model\_four\_spheres.py_
simulate a sequence of CSD profiles to predict potentials on the electrodes.


| argument         | description                                                       |
|------------------|-------------------------------------------------------------------|
| `--output`       | path to the _CSV_ file where the potentials are to be stored      |
| `--sources`      | path to the file with volumetric CSD tensor (4D array)            |
| `--config`       | path to the model config file                                     |
| `--electrodes`   | path to the electrode location file                               |
| `--name`         | name of the electrode                                             |
| `--mesh`         | path to the main _FEniCS_ mesh                                    |
| `--degree`       | degree of the FEM element (defaults to 1)                         |
| `--element-type` | type of the FEM element (defaults to `'CG'` (Continuous Galerkin) |
| `--quiet`        | supress control messages                                          |
| `--start-from`   | number of the first source to start from (defaults to 0)          |

The `--start-from` argument is useful in case of a broken run, as the output
file is the electrode location file with additional fields, and it is saved
after simulating of each source.  Thus in case of a broken run it may be given
as `--electrodes` with request to complete simulation of missing sources.


#### Slice specific

The `--ground-potential` parameter is the potantial \[ $V$ \] at the grounded
slice-covering dome.  If not given a $0 V$ grounding is assumed at infinity.


#### Sphere specific

The `--grounded-plate-edge-z` parameter is the Z coordinate \[ $m$ \] of the
edge of the grounded ($0 V$ potential) conductive plate.  Defaults to $-88 mm$.
