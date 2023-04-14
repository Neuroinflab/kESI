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
| `csd_basis_functions` | [CSD basis functions](#data-bundled-csd_basis_functions)                                                                      |
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
in the _gmsh_ geometry format.  Here, `circular_slice` is the `<geometry>`
wildcard, and `composite` is the `<granularity>`
(see [section regarding generated meshes](#data-generated-meshes) for details).


#### Mesh geometry templates <a name="data-bundled-mesh_geometry_templates"></a>

Filenames of templates of _gmsh_ geometry files follow the `<mesh stem>.geo.template`
pattern, where `<mesh stem>` determines most mesh properties but
size of its particular elements (which is controlled by the
`SED_RELATIVE_ELEMENT_SIZE` template marker).


#### Model properties <a name="data-bundled-model_properties"></a>

Physical properties of the model (i.e. medium conductivity) and
its geometrical properties (e.g. sphere radius) are stored in the
`<model>.ini` files, where `<model>` follows `<geometry>__<conductivity>`
pattern.  `<geometry>` itself may be also `four_spheres_csf_1_mm__separate_cortex`,
where `four_spheres_csf_1_mm` refers the proper geometry and `separate_cortex`
to its particular subpartitioning in _gmsh_ mesh.


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
  <mesh stem>/
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
- `<mesh stem>` is the stem of the
  [mesh geometry template file](#data-bundled-mesh_geometry_templates)
  and follows the `<geometry>[__<version>]` pattern (part in the
  `[]` is optional),
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

The `electrodes.csv` files contain a table  with location of the electrodes,
with columns named intuitively: _NAME_ (name of the electrode), _X_, _Y_ and _Z_
(its location coordinates in meters).  An examplary file may look like:
```
NAME,X,Y,Z
A_00,-0.006,0.0,0.046
A_01,-0.006,0.0,0.0485
```

The filesystem subtree follows the pattern:
```
setups/
  [<aaa>__[<bbb>__[...]]]<zzz>/
    electrodes.csv
```
(part in `[]` is optional and `...` is used instead of further recursion),
where directory name encodes path to [the bundled file](#data-bundled-position_of_electrodes)
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

The filesystem subtree follows the pattern:
```
fenics_leadfield_corrections/
  <setup>/
    <model>/
      conductivity.ini
      <mesh path>/
        <degree>/
          <electrode>.h5
          <electrode>.ini
```
where:
- `<setup>` wildcard was defined in the [Setups](#data-generated-setups) subsection,
- `<model>` was defined in [Model properties](#data-bundled-model_properties),
- `<mesh path>` is either `<granularity>` or `<version>__<granularity>`
  (see the [Meshes](#data-generated-meshes) subsection for details),
- `<degree>` is the degree of elements used by the finite element method (FEM)
  to calculate the correction,
- `<electrode>` is the name of the electrode in the setup.

The `<electrode>.h5` file contains correction of the leadfield of the electrode
`<electrode>` saved as a 3D _FEniCS_ scaler function $[V/A]$,
and file `<electrode>.ini` contains its metadata:

| section    | field                     | value                                                       |
|------------|---------------------------|-------------------------------------------------------------|
| fem        | mesh                      | path to the main mesh file                                  |
|            | degree                    | degree of the element                                       |
|            | element_type              | type of the element                                         |
| model      | config                    | path to model properties (conductivity etc.)                |
| electrode  | x                         | X coordinate of the point electrode $[m]$                   |
|            | y                         | Y coordinate of the point electrode $[m]$                   |
|            | z                         | Z coordinate of the point electrode $[m]$                   |
| correction | global_preprocessing_time | location-independent preprocessing time $[s]$               |
|            | setup_time                | total function manager and FEM initiation time $[s]$        |
|            | total_solving_time        | total time of location-dependent processing $[s]$           |
|            | local_preprocessing_time  | location-dependent preprocessing time $[s]$                 |
|            | solving_time              | time of FEM equation solving $[s]$                          |
|            | base_conductivity         | base conductivity used by renormalization $[S/m]$           |
|            | filename                  | relative path to the correction function (`<electrode>.h5`) |

The `conductivity.ini` file is a copied [Model properties](#data-bundled-model_properties)
file appropriate for the `<model>`.  Note that the file **may contain also geometrical
data**.


#### Sampled leadfield corrections <a name="data-generated-sampled_leadfield_corrections"></a>

The filesystem subtree follows the pattern:
```
sampled_leadfield_corrections/
  <setup>/
    <sampling>/
      grid.npz
      <model>/
        <mesh path>/
          <degree>/
            <electrode>.npz
```
where `<sampling>` defines the regular grid the leadfield correction
function is sampled on and other wildcards were discussed in
[the previous section](#data-generated-fenics_leadfield_corrections).

<a name="data-generated-sampled_leadfield_corrections-grid_npz"></a>
The `grid.npz` is a compressed NumPy file containing the $n^{POT}_x \times n^{POT}_y \times n^{POT}_z$
regular grid description:

| array name | shape                         | type          | content                            |
|------------|-------------------------------|---------------|------------------------------------|
| `X`        | $n^{POT}_x \times 1 \times 1$ | `float` $[m]$ | grid nodes projected on the X axis |
| `Y`        | $1 \times n^{POT}_y \times 1$ | `float` $[m]$ | grid nodes projected on the Y axis |
| `Z`        | $1 \times 1 \times n^{POT}_z$ | `float` $[m]$ | grid nodes projected on the Z axis |

The grid is a Cartesian product of `X`, `Y` and `Z` arrays, which may be obtained with a call
to `numpy.meshgrid(X, Y, Z, indexing='ij')`.

The compressed NumPy file `<electrode>.npz` contains the sampled leadfield correction
with additional (meta)data:

| array name             | shape                                         | type            | content                                                             |
|------------------------|-----------------------------------------------|-----------------|---------------------------------------------------------------------|
| `CORRECTION_POTENTIAL` | $n^{POT}_x \times n^{POT}_y \times n^{POT}_z$ | `float` $[V/A]$ | sampled leadfield correction                                        |
| `X`                    | $n^{POT}_x \times 1 \times 1$                 | `float` $[m]$   | grid nodes projected on the X axis                                  |
| `Y`                    | $1 \times n^{POT}_y \times 1$                 | `float` $[m]$   | grid nodes projected on the Y axis                                  |
| `Z`                    | $1 \times 1 \times n^{POT}_z$                 | `float` $[m]$   | grid nodes projected on the Z axis                                  |
| `LOCATION`             | $3$                                           | `float` $[m]$   | X, Y, Z coordinates of the electrode                                |
| `BASE_CONDUCTIVITY`    | scalar                                        | `float` $[S/m]$ | medium conductivity for which the corrected potential is calculated |
| `_PREPROCESSING_TIME`  | scalar                                        | `float` $[s]$   | construction time of the `FunctionManager` object                   |
| `_LOADING_TIME`        | scalar                                        | `float` $[s]$   | loading time of the leadfield correction function                   |
| `_PROCESSING_TIME`     | scalar                                        | `float` $[s]$   | leadfield correction sampling time                                  |
| `_R_LIMIT`             | $2$                                           | `float` $[m]$   | inclusive limits of sampling radius (spherical sampling only)       |

[//]: # (TODO: explain samplings: romberg_k; cropped_*)


#### Potential basis functions at electrodes <a name="data-generated-potential_basis_functions_at_electrodes"></a>

The filesystem subtree follows the pattern:
```
potential_basis_functions/
  <setup>/
    <csd basis functions>/
      centroids.npz
      model_src.json
      <inverse model path>/
        <electrode>.npz
```
where `<csd basis functions>` defines shape and location of basis functions
in the CSD codomain and `<inverse model path>` defines model (computational
as well as physical) used to couple them with their counterparts in the potential
codomain.  The `<setup>` wildcard was defined in the [Setups](#data-generated-setups)
subsection; and `<electrode>`, in the [Fenics leadfield corrections](#data-generated-fenics_leadfield_corrections).

The shape of the basis functions radially spline-defined in the CSD codomain
is defined in the `model_src.json` function:

```python
model_src = SphericalSplineSourceBase.fromJSON(open('model_src.json'))
```

Locations of their centroids (coordinate system origins) are defined
in the compressed NumPy file `centroids.npz`:

| array name | shape                                         | type          | content                            |
|------------|-----------------------------------------------|---------------|------------------------------------|
| `X`        | $n^{SRC}_x \times 1 \times 1$                 | `float` $[m]$ | grid nodes projected on the X axis |
| `Y`        | $1 \times n^{SRC}_y \times 1$                 | `float` $[m]$ | grid nodes projected on the Y axis |
| `Z`        | $1 \times 1 \times n^{SRC}_z$                 | `float` $[m]$ | grid nodes projected on the Z axis |
| `MASK`     | $n^{SRC}_x \times n^{SRC}_y \times n^{SRC}_z$ | `bool`        | subset of grid nodes               |

The Cartesian product of `X`, `Y` and `Z` defines regular grid on which the centroids may
be located, while `MASK` defines its ordered subset (the locations of centroids):

```python
GRID = numpy.meshgrid(X, Y, Z, indexing='ij')
CENTROIDS = [A[MASK] for A in GRID]
```

For kCSD `<inverse model path>` is `kCSD/<conductivity [S/m]>`,
where `<conductivity [S/m]>` is the assumed (scalar) medium conductivity
in an appropriate SI unit.
For kESI `<inverse model path>` is `kESI/<sampling>/<model>/<mesh path>/<degree>/`,
where `<sampling>` wildcard was defined in [the previous section](#data-generated-sampled_leadfield_corrections);
and other wildcards, in [Fenics leadfield corrections](#data-generated-fenics_leadfield_corrections).

<a name="data-generated-potential_basis_functions_at_electrodes-electrodes_npz"></a>
The `<electrode>.npz` contains values of basis functions at the location
of the electrode in the potential codomain ($\phi$ function - $\Phi$ in
[Chintaluri 2021](#bibliography-chintaluri2021)):

| array name     | shape  | type            | content                                 |
|----------------|--------|-----------------|-----------------------------------------|
| `POTENTIALS`   | $M$    | `float` $[V]$   | subset of grid nodes                    |
| `X`            | scalar | `float` $[m]$   | X coordinate of the electrode           |
| `Y`            | scalar | `float` $[m]$   | Y coordinate of the electrode           |
| `Z`            | scalar | `float` $[m]$   | Z coordinate of the electrode           |
| `CONDUCTIVITY` | scalar | `float` $[S/m]$ | assumed medium conductivity (kCSD only) |


#### Kernels <a name="data-generated-kernels"></a>

The filesystem subtree follows the pattern:
```
kernels/
  <setup>/
    <subsetup>/
      electrodes.csv
      <csd basis functions>/
        <inverse model path>/
          phi.npz
          kernel.npz
          analysis.npz
          <csd grid>/
            grid_csd.npz
            crosskernel.npz
```
where `<subsetup>` defines an ordered subset of electrodes from `<setup>`,
for which the kernel is calculated, and `<csd grid>` defines
the grid of CSD estimation points.  The other wildcards were discussed
extensively in previous sections.

File `electrodes.csv` contains a header and $N$ rows (which order defines
order of electrodes in the `<subsetup>`).  Every row contains (redundant)
electrode location as well as its name.

| field  | type          | content               |
|--------|---------------|-----------------------|
| `NAME` | `str`         | name of the electrode |
| `X`    | `float` $[m]$ | X coordinate of ...   |
| `Y`    | `float` $[m]$ | Y coordinate of ...   |
| `Z`    | `float` $[m]$ | Z coordinate of ...   |

The compressed NumPy file `phi.npz` contains a single array `PHI`
of shape $M \times N$ and type `float` $[V]$, which columns are
locations of electrodes mapped according to the $\phi$ function
($\Phi$ in [Chintaluri 2021](#bibliography-chintaluri2021)):

$$
\Phi = \left[ \phi(e_1), \phi(e_2), \dots, \phi(e_{M-1}), \phi(e_M) \right]
$$

The columns of the $\Phi$ matrix are `POTENTIALS` vectors from the
[`<electrode>.npy` files described in the previous section](#data-generated-potential_basis_functions_at_electrodes-electrodes_npz)
for details).

The compressed NumPy file `kernel.npz` contains a single array `KERNEL`
of shape $N \times N$ and type `float` $[V^2]$, which is _kCSD_/_kESI_
kernel matrix:

$$
\mathbf{K} = \Phi^T \Phi
$$


The compressed NumPy file `analysis.npz` contains auxiliary analytical
data.

| array            | shape        | type            | content                                                  |
|------------------|--------------|-----------------|----------------------------------------------------------|
| `EIGENVALUES`    | $N$          | `float` $[V^2]$ | Kernel eigenvalues (diagonal of the $\Lambda$ matrix)    |
| `EIGENSOURCES`   | $M \times N$ | `float` $[1]$   | Eigensources in the canonical form ($\mathbf{U})$        |
| `SINGULARVALUES` | $N$          | `float` $[V]$   | $\Phi$ singular values (diagonal of the $\Sigma$ matrix) |
| `EIGENVECTORS`   | $N \times N$ | `float` $[1]$   | Kernel eigenvalues ($\mathbf{W}$)                        |

where $\mathbf{U}$, $\Sigma$, $\mathbf{W}$ are singular value decomposition of $\Phi$:

$$
\Phi = \mathbf{U} \Sigma \mathbf{W}^T
\text{,}
$$

$\mathbf{W}$ and $\Lambda = \Sigma^2$ are eigendecomposition of $\mathbf{K}$:

$$
\mathbf{K} = \mathbf{W} \Lambda \mathbf{W}^T
\text{.}
$$

Note that according to notation from [Chintaluri 2021](#bibliography-chintaluri2021)

$$
\mathbf{W} = \left[ \mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_{N-1}, \mathbf{w}_N \right]
\text{,}
$$

and

$$
\Lambda_{i, j} = \begin{cases}
  \mu_i & \text{if } i = j \text{,} \\
  0 & \text{otherwise.}
\end{cases}
$$


The `grid_csd.npz` file defines the grid on which the CSD is estimated.
It is similar to
[the previously described `grid.npz` file](#data-generated-sampled_leadfield_corrections-grid_npz).
The only difference is $CSD$ substituting the $POT$ as a $n$ superscript index.

[//]: # (TODO)

[//]: # (            crosskernel.npz)




#### CSD profiles <a name="data-generated-csd_profiles"></a>

The filesystem subtree follows the pattern:
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

Directories following pattern
_images/\<geometry\>/\<granularity\>/\<degree\>/kernels/\<k\>/_
contain kESI/kCSD kernel-related files derived from
_\<geometry\>/\<granularity\>/\<degree\>/sampled/\<k\>/*.npz_
sampled leadfields.

| file                        | content                                                        |
|-----------------------------|----------------------------------------------------------------|
| `electrodes.csv`            | positions of electrodes                                        |
| `src_mask.npz`              | positions of source centroids                                  |
| `<method>_phi.npz`          | the transfer matrix ($\Phi$) of `<method>`                     |
| `<method>_kernel.npz`       | the kernel matrix of `<method>`                                |
| `<method>_crosskernel.npz`  | the volumetric crosskernel tensor of `<method>`                |
| `<method>_analysis.npz`     | auxilary analytical data of `<method>`                         |
| `<method>_eigensources.npz` | the volumetric eigensource tensor of `<method>`                |
| `fair_sources.npz`          | average of appropriate volumetric eigensources of both methods |

`<method>` may be either `kCSD` or `kESI`.  Note that
`images/<geometry>/<granularity>/<degree>/kernels/<k>/kCSD_*.npz`
files are redundant.

In the `images/<geometry>/<granularity>/<degree>/kernels/<k>/images`
subtree (`<inverse model>`) results of the forward modelling are stored.  File
`<inverse model>/<geometry>/<granularity>/<degree>/<sources>.csv`
contains potentials at the electrodes generated by CSD profiles from
`<inverse model>/../<sources>.npz` with a FEM forward model appropriate
for the `<geometry>/<granularity>/<degree>` subpath.


## Files


### Volumetric cross-kernel tensor

A compressed NumPy file (`*.npz`).

| array         | shape                                                  | type                | content                                 |
|---------------|--------------------------------------------------------|---------------------|-----------------------------------------|
| `CROSSKERNEL` | $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times n$ | `float` $[W / m^3]$ | The crosskernel tensor ($\overline{K}$) |
| `X`           | $n^{CSD}_x \times 1 \times 1$                          | `float` $[m]$       | X nodes of the CSD sampling grid        |
| `Y`           | $1 \times n^{CSD}_y \times 1$                          | `float` $[m]$       | Y nodes of the CSD sampling grid        |
| `Z`           | $1 \times 1 \times n^{CSD}_z$                          | `float` $[m]$       | Z nodes of the CSD sampling grid        |

The tensor yields volumetric CSD reconstruction $C = \overline{K} K^{-1} V$,
where $V$ is a vector (matrix in case of timepoints) of measured potentials and
$C$ is an $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z$ array.


### Volumetric eigensource tensor

A compressed NumPy file (`*.npz`).

| array | shape                                                  | type                | content                          |
|-------|--------------------------------------------------------|---------------------|----------------------------------|
| `CSD` | $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times n$ | `float` $[A / m^3]$ | sampled CSDs of n eigensources   |
| `X`   | $n^{CSD}_x \times 1 \times 1$                          | `float` $[m]$       | X nodes of the CSD sampling grid |
| `Y`   | $1 \times n^{CSD}_y \times 1$                          | `float` $[m]$       | Y nodes of the CSD sampling grid |
| `Z`   | $1 \times 1 \times n^{CSD}_z$                          | `float` $[m]$       | Z nodes of the CSD sampling grid |


### Potentials at electrodes generated by CSD profiles

A CSV file.

| field        | type          | content                                                   |
|--------------|---------------|-----------------------------------------------------------|
| `NAME`       | `str`         | name of the electrode                                     |
| `X`          | `float` $[m]$ | X position of the electrode                               |
| `Y`          | `float` $[m]$ | Y position of the electrode                               |
| `Z`          | `float` $[m]$ | Z position of the electrode                               |
| `SOURCE_<i>` | `float` $[V]$ | potential generated by the `i`-th source at the electrode |


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

The `--ground-potential` parameter is the potantial $[V]$ at the grounded
slice-covering dome.  If not given a $0 V$ grounding is assumed at infinity.


#### Sphere specific

The `--grounded-plate-edge-z` parameter is the Z coordinate $[m]$ of the
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

The `--sampling-radius` parameter is the edge length $[m]$ of the sampled
cube.  It defaults to $0.3 mm$.

The cube is based at the $Z = 0$ plane and centered at $X = Y = 0$ axis.


#### Sphere specific

The `--sampling-radius` parameter is the radius $[m]$ of the sampled sphere
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

The `--ground-potential` parameter is the potantial $[V]$ at the grounded
slice-covering dome.  If not given a $0 V$ grounding is assumed at infinity.


#### Sphere specific

The `--grounded-plate-edge-z` parameter is the Z coordinate $[m]$ of the
edge of the grounded ($0 V$ potential) conductive plate.  Defaults to $-88 mm$.


## Bibliography

1. <a name="bibliography-chintaluri2021"></a>Chaitanya Chintaluri, Marta Bejtka,
   Władysław Średniawa, Michał Czerwiński, Jakub M. Dzik, Joanna Jędrzejewska-Szmek,
   Kacper Kondrakiewicz, Ewa Kublik, Daniel K. Wójcik,
   **What we can and what we cannot see with extracellular multielectrodes**,
   2021, _PLoS Computational Biology_ 17(5): e1008615,
   DOI: [10.1371/journal.pcbi.1008615](https://doi.org/10.1371/journal.pcbi.1008615)
2. <a name="bibliography-zahn1989"></a>Markus Zahn, **Pole elektromagnetyczne**,
   1989 Warszawa, Państwowe Wydawnictwo Naukowe, ISBN: 83-01-07693-3
   (original title: Electromagnetic Field Theory: a problem solving approach)
