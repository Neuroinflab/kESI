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

##### File `electrodes.csv` <a name="data-generated-setups-electrodes_csv"></a>

The `electrodes.csv` files contain a table  with location of the electrodes,
with columns named intuitively: _NAME_ (name of the electrode), _X_, _Y_ and _Z_
(its location coordinates in meters).  An exemplary file may look like:
```
NAME,X,Y,Z
A_00,-0.006,0.0,0.046
A_01,-0.006,0.0,0.0485
```

| field  | type          | content               |
|--------|---------------|-----------------------|
| `NAME` | `str`         | name of the electrode |
| `X`    | `float` $[m]$ | X coordinate of ...   |
| `Y`    | `float` $[m]$ | Y coordinate of ...   |
| `Z`    | `float` $[m]$ | Z coordinate of ...   |


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

##### Files `<electrode>.h5` and `<electrode>.ini` <a name="data-generated-fenics_leadfield_corrections-electrode"></a>

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


##### File `conductivity.ini` <a name="data-generated-fenics_leadfield_corrections-conductivity_ini"></a>

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

##### File `grid.npz` <a name="data-generated-sampled_leadfield_corrections-grid_npz"></a>

The `grid.npz` is a compressed NumPy file containing the $n^{POT}_x \times n^{POT}_y \times n^{POT}_z$
regular grid description:

| array name | shape                         | type          | content                            |
|------------|-------------------------------|---------------|------------------------------------|
| `X`        | $n^{POT}_x \times 1 \times 1$ | `float` $[m]$ | grid nodes projected on the X axis |
| `Y`        | $1 \times n^{POT}_y \times 1$ | `float` $[m]$ | grid nodes projected on the Y axis |
| `Z`        | $1 \times 1 \times n^{POT}_z$ | `float` $[m]$ | grid nodes projected on the Z axis |

The grid is a Cartesian product of `X`, `Y` and `Z` arrays, which may be obtained with a call
to `numpy.meshgrid(X, Y, Z, indexing='ij')`.


##### File `<electrode>.npz` <a name="data-generated-sampled_leadfield_corrections-electrode_npz"></a>

The compressed NumPy file `<electrode>.npz` contains the sampled leadfield correction
with additional (meta)data:

| array name             | shape                                         | type            | content                                                             |
|------------------------|-----------------------------------------------|-----------------|---------------------------------------------------------------------|
| `CORRECTION_POTENTIAL` | $n^{POT}_x \times n^{POT}_y \times n^{POT}_z$ | `float` $[V/A]$ | sampled leadfield correction                                        |
| `X`                    | $n^{POT}_x \times 1 \times 1$                 | `float` $[m]$   | grid nodes projected on the X axis (redundant with `grid.npz` file) |
| `Y`                    | $1 \times n^{POT}_y \times 1$                 | `float` $[m]$   | grid nodes projected on the Y axis (redundant with `grid.npz` file) |
| `Z`                    | $1 \times 1 \times n^{POT}_z$                 | `float` $[m]$   | grid nodes projected on the Z axis (redundant with `grid.npz` file) |
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
      model_src.json
      centroids.npz
      <inverse model path>/
        <electrode>.npz
```
where `<csd basis functions>` defines shape and location of basis functions
in the CSD codomain and `<inverse model path>` defines model (computational
as well as physical) used to couple them with their counterparts in the potential
codomain.  The `<setup>` wildcard was defined in the [Setups](#data-generated-setups)
subsection; and `<electrode>`, in the [Fenics leadfield corrections](#data-generated-fenics_leadfield_corrections).


##### File `model_src.json` <a name="data-generated-potential_basis_functions_at_electrodes-model_src_json"></a>

The shape of the basis functions radially spline-defined in the CSD codomain
is defined in the `model_src.json` function:

```python
model_src = SphericalSplineSourceBase.fromJSON(open('model_src.json'))
```


##### File `centroids.npz` <a name="data-generated-potential_basis_functions_at_electrodes-centroids_npz"></a>

Locations of their centroids (coordinate system origins) are defined
in the compressed NumPy file `centroids.npz`:

| array name | shape                                         | type          | content                            |
|------------|-----------------------------------------------|---------------|------------------------------------|
| `X`        | $n^{SRC}_x \times 1 \times 1$                 | `float` $[m]$ | grid nodes projected on the X axis |
| `Y`        | $1 \times n^{SRC}_y \times 1$                 | `float` $[m]$ | grid nodes projected on the Y axis |
| `Z`        | $1 \times 1 \times n^{SRC}_z$                 | `float` $[m]$ | grid nodes projected on the Z axis |
| `MASK`     | $n^{SRC}_x \times n^{SRC}_y \times n^{SRC}_z$ | `bool`        | subset of grid nodes               |

The Cartesian product of `X`, `Y` and `Z` defines regular grid on which
the centroids may be located, while `MASK` defines its ordered subset
(the locations of centroids):

```python
GRID = numpy.meshgrid(X, Y, Z, indexing='ij')
CENTROIDS = [A[MASK] for A in GRID]
```

For kCSD `<inverse model path>` is `kCSD/<conductivity>`, where `<conductivity>`
is the assumed (scalar) medium conductivity in an appropriate SI unit ($S/m$).
For kESI `<inverse model path>` is `kESI/<sampling>/<model>/<mesh path>/<degree>/`,
where `<sampling>` wildcard was defined in
[the previous section](#data-generated-sampled_leadfield_corrections);
and other wildcards, in [Fenics leadfield corrections](#data-generated-fenics_leadfield_corrections).


##### File `<electrode>.npz` <a name="data-generated-potential_basis_functions_at_electrodes-electrodes_npz"></a>

The `<electrode>.npz` contains values of basis functions at the location
of the electrode in the potential codomain ($\mathbf{b}$ vector function
- $\Phi$ in [Chintaluri 2021](#bibliography-chintaluri2021)):

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
          potential_basis_functions.npz
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

##### File `electrodes.csv` <a name="data-generated-kernels-electrodes_csv"></a>

File `electrodes.csv` contains a header and $N$ rows (which order defines
order of electrodes in the `<subsetup>`).  Every row contains (redundant -
see information in
[the `electrodes.csv` in the Setups section](#data-generated-setups-electrodes_csv))
electrode location as well as its name.

| field  | type          | content               |
|--------|---------------|-----------------------|
| `NAME` | `str`         | name of the electrode |
| `X`    | `float` $[m]$ | X coordinate of ...   |
| `Y`    | `float` $[m]$ | Y coordinate of ...   |
| `Z`    | `float` $[m]$ | Z coordinate of ...   |


##### File `potential_basis_functions.npz` <a name="data-generated-kernels-pbf"></a>

The compressed NumPy file `potential_basis_functions.npz` contains a single array `B`
of shape $M \times N$ and type `float` $[V]$, which columns are
locations of electrodes mapped according to the $\mathbf{b}$ function
($\Phi$ in [Chintaluri 2021](#bibliography-chintaluri2021)):

$$
\mathbf{B} = \left[ \mathbf{b}(e_1), \mathbf{b}(e_2), \dots, \mathbf{b}(e_{M-1}), \mathbf{b}(e_M) \right]
$$

The columns of the $\mathbf{B}$ matrix are `POTENTIALS` vectors from the
[`<electrode>.npy` files described in the previous section](#data-generated-potential_basis_functions_at_electrodes-electrodes_npz)
for details).


##### File `kernel.npz` <a name="data-generated-kernels-kernel_npz"></a>

The compressed NumPy file `kernel.npz` contains a single array `KERNEL`
of shape $N \times N$ and type `float` $[V^2]$, which is _kCSD_/_kESI_
kernel matrix:

$$
\mathbf{K} = \mathbf{B}^T \mathbf{B}
$$


##### File `analysis.npz` <a name="data-generated-kernels-analysis_npz"></a>

The compressed NumPy file `analysis.npz` contains auxiliary analytical
data.

| array            | shape        | type            | content                                                        |
|------------------|--------------|-----------------|----------------------------------------------------------------|
| `EIGENVALUES`    | $N$          | `float` $[V^2]$ | Kernel eigenvalues (diagonal of the $\Lambda$ matrix)          |
| `EIGENSOURCES`   | $M \times N$ | `float` $[1]$   | Eigensources in the canonical form ($\mathbf{U})$              |
| `SINGULARVALUES` | $N$          | `float` $[V]$   | $\mathbf{B}$ singular values (diagonal of the $\Sigma$ matrix) |
| `EIGENVECTORS`   | $N \times N$ | `float` $[1]$   | Kernel eigenvalues ($\mathbf{W}$)                              |

where $\mathbf{U}$, $\Sigma$, $\mathbf{W}$ are singular value decomposition of $\mathbf{B}$:

$$
\mathbf{B} = \mathbf{U} \Sigma \mathbf{W}^T
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


##### File `grid_csd.npz` <a name="data-generated-kernels-grid_csd_npz"></a>

The `grid_csd.npz` file defines the grid on which the CSD is estimated.
It is similar to
[the previously described `grid.npz` file](#data-generated-sampled_leadfield_corrections-grid_npz).
The only difference is $CSD$ substituting the $POT$ as a $n$ superscript index.


##### File `crosskernel.npz` <a name="data-generated-kernels-crosskernel_npz"></a>

The `crosskernel.npz` is a compressed NumPy file, which contains
(redundant - see
[the `grid_csd.npz` file](#data-generated-kernels-grid_csd_npz)
for details) the grid on which the CSD is estimated, as well as
the crosskernel tensor $\mathbf{\tilde{K}}$.  The tensor yields
volumetric CSD reconstruction

$$
\mathbf{C}^* = \mathbf{\tilde{K}} \mathbf{K}^{-1} \mathbf{V}
\text{,}
$$

where $\mathbf{V}$ is either an $N$-element vector or $N \times n_t$
matrix (with $n_t$ timepoints) of measured potentials, and $\mathbf{C}^*$
is either an $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z$
or $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times n_t$ array,
respectively.

| array         | shape                                                  | type                | content                              |
|---------------|--------------------------------------------------------|---------------------|--------------------------------------|
| `CROSSKERNEL` | $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times N$ | `float` $[W / m^3]$ | The crosskernel tensor ($\tilde{K}$) |
| `X`           | $n^{CSD}_x \times 1 \times 1$                          | `float` $[m]$       | X nodes of the CSD sampling grid     |
| `Y`           | $1 \times n^{CSD}_y \times 1$                          | `float` $[m]$       | Y nodes of the CSD sampling grid     |
| `Z`           | $1 \times 1 \times n^{CSD}_z$                          | `float` $[m]$       | Z nodes of the CSD sampling grid     |

Note that crosskernel $\mathbf{\tilde{K}} ~ [W / m^3]$
is multiplied by $\beta = \mathbf{K}^{-1}\mathbf{V} ~ [V^{-1}]$,
thus the unit of $\mathbf{C}^* = \mathbf{\tilde{K}} \beta$ is $[A / m^3]$.

[//]: # (TODO: relation between CSD, POT and SRC - centroid - grids)




#### CSD profiles <a name="data-generated-csd_profiles"></a>

The filesystem subtree follows the pattern:
```
csd_profiles/
  <setup>/
    <subsetup>/
      <csd basis functions>/
        <csd path>
          <csd_grid>
            <profiles>.npz
            <fwd path>/
              <profiles>.csv
```
where `<profiles>` defines CSD profiles; and `<csd path>`, their provenance.
The `<csd path>` may be either `<inverse model path>`
or `mixed/<conductivity>/<kESI path>`, where `<kESI path>` stands for
`<sampling>/<model>/<mesh path>/<degree>`.

If `<profiles>` are `eigensources` and `<csd path>` is:
- `<inverse model path>` then CSD profiles are eigensources of either
  appropriate inverse model (with regard to the `<inverse model path>`),
- `mixed/<conductivity>/<kESI path>`, then CSD profiles are averaged
  (matching) eigensources for `kCSD/<conductivity>`
  and `kESI/<kESI path>`.

The `<fwd path>` is `<fwd model>/<fwd mesh path>/<fwd degree>`. It determines
the FEM (_FEniCS_) forward model used to simulate the potential generated
by a CSD profiles in a way similar to described in the
[Fenics leadfield corrections](#data-generated-fenics_leadfield_corrections)
section, that is:
- `<fwd model>` was defined in [Model properties](#data-bundled-model_properties),
- `<fwd mesh path>` is either `<granularity>` or `<version>__<granularity>`
  (see the [Meshes](#data-generated-meshes) subsection for details),
- `<fwd degree>` is the degree of elements used by the finite element method (FEM).

All other wildcards were discussed in previous sections.

##### File `<profiles>.npz` <a name="data-generated-csd_profiles-profiles_npz"></a>

The `<profiles>.npz` is a compressed NumPy file, which contains
(possibly redundant - see
[the `grid_csd.npz` file](#data-generated-kernels-grid_csd_npz)
for details) the grid on which the CSD is estimated, as well as
$n_{CSD}$ volumetric CSD profiles (3D arrays of
$n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z$ shape stacked
along the last axis).

| array | shape                                                        | type                | content                          |
|-------|--------------------------------------------------------------|---------------------|----------------------------------|
| `CSD` | $n^{CSD}_x \times n^{CSD}_y \times n^{CSD}_z \times n_{CSD}$ | `float` $[A / m^3]$ | CSD profiles                     |
| `X`   | $n^{CSD}_x \times 1 \times 1$                                | `float` $[m]$       | X nodes of the CSD sampling grid |
| `Y`   | $1 \times n^{CSD}_y \times 1$                                | `float` $[m]$       | Y nodes of the CSD sampling grid |
| `Z`   | $1 \times 1 \times n^{CSD}_z$                                | `float` $[m]$       | Z nodes of the CSD sampling grid |

If `<profiles>` are `eigensources` then $n_{CSD} = N$.


##### File `<profiles>.csv` <a name="data-generated-csd_profiles-profiles_csv"></a>

File `<profiles>.csv` contains a header and $N$ rows.
Every row contains (redundant - see information in
[the `electrodes.csv` in the Setups section](#data-generated-setups-electrodes_csv))
electrode location as well as its name and simulated
potential values for $n_{CSD}$ CSD profiles.

| field           | type          | content                          |
|-----------------|---------------|----------------------------------|
| `NAME`          | `str`         | name of the electrode            |
| `X`             | `float` $[m]$ | X coordinate of ...              |
| `Y`             | `float` $[m]$ | Y coordinate of ...              |
| `Z`             | `float` $[m]$ | Z coordinate of ...              |
| `POTENTIAL_<i>` | `float` $[V]$ | potential for $i$-th CSD profile |


## Tools

### Leadfield correction solving

`solve_slice_on_plate.py` and `solve_sphere_on_plate.py`
calculate
[the leadfield correction _FEniCS_ function](#data-generated-fenics_leadfield_corrections-electrode)
for slice and spherical models, respectively.
For that purpose [mesh](#data-generated-meshes),
[electrode location](#data-generated-setups-electrodes_csv) and
[model properties](#data-generated-fenics_leadfield_corrections-conductivity_ini)
are taken as input.


### Model source generation

`create_model_src.py` generates a spherical model source
of a given size.  Relation between CSD and distance from
the centroid is given by a sigmoid spline function.


### Leadfield correction sampling

`create_grid.py` generates [the sampling grid](#data-generated-sampled_leadfield_corrections-grid_npz) used by either
`sample_volumetric_solution.py` or `sample_spherical_solution.py`
to sample [_FEniCS_ functions](#data-generated-fenics_leadfield_corrections-electrode)
for either slice or spherical model
as [3D arrays of values](#data-generated-sampled_leadfield_corrections-electrode_npz).

`crop_grid_to_sources.py` crops
[the sampling grid](#data-generated-sampled_leadfield_corrections-grid_npz)
to a minimal grid necessary for
[a CSD profile](#data-generated-potential_basis_functions_at_electrodes-model_src_json)
centered at
[centroid nodes](#data-generated-potential_basis_functions_at_electrodes-centroids_npz).


### Potential basis functions vector calculation

`calculate_kcsd_potential_basis_function.py` creates
[a vector of values of potential basis functions](#data-generated-potential_basis_functions_at_electrodes-electrodes_npz)
for kCSD assumptions taking
[a CSD profile](#data-generated-potential_basis_functions_at_electrodes-model_src_json),
[centroid nodes](#data-generated-potential_basis_functions_at_electrodes-centroids_npz),
medium conductivity and [electrode location](#data-generated-setups-electrodes_csv).

In contrast, `calculate_kesi_potential_basis_function.py`
calculates the kESI-corrected vector taking
[the sampled leadfield correction](#data-generated-sampled_leadfield_corrections-electrode_npz)
instead of conductivity and electrode location
(which are included therein).


### Kernel calculation

`calculate_kernel.py` generates:
- [an array of values of potential basis functions at locations of electrodes](#data-generated-kernels-pbf),
- [auxiliary analytical data](#data-generated-kernels-analysis_npz),
- [a kernel itself](#data-generated-kernels-kernel_npz).
For that purpose
[an ordered list of electrodes](#data-generated-kernels-electrodes_csv)
and appropriate
[vectors of values of potential basis functions](#data-generated-potential_basis_functions_at_electrodes-electrodes_npz)
are taken as input.


### Crosskernel calculation

`calculate_volumetric_crosskernel.py` calculates
[a volumetric crosskernel](#data-generated-kernels-crosskernel_npz) using:
- [an array of values of potential basis functions at locations of electrodes](#data-generated-kernels-pbf),
- [a CSD profile](#data-generated-potential_basis_functions_at_electrodes-model_src_json),
- [centroid nodes](#data-generated-potential_basis_functions_at_electrodes-centroids_npz),
- [CSD estimation grid](#data-generated-kernels-grid_csd_npz).


### Volumetric CSD profiles calculation

There are two tools calculating
[volumetric CSD profiles](#data-generated-csd_profiles-profiles_npz):
- `calculate_volumetric_eigensources.py` calculates
  eigensources from
  [auxiliary analytical data](#data-generated-kernels-analysis_npz)
  and
  [centroid nodes](#data-generated-potential_basis_functions_at_electrodes-centroids_npz).
  CSD profiles are sampled at a
  [given CSD grid](#ata-generated-kernels-grid_csd_npz).

- `mix_eigensources.py` averages two sets of volumetric CSD profiles
  of eigensources.  Before averaging eigensources are matched based on appropriate
  [auxiliary analytical data](#data-generated-kernels-analysis_npz) files.
  The match is based on the absolute value of the dot product of eigensource
  canonical form.

[//]: # (Note that both sets must share at least shape of the matrix $\mathbf{B}$.)


### Forward modelling

[Potential values](#data-generated-csd_profiles-profiles_csv)
at given [electrode locations](#data-generated-kernels-electrodes_csv)
are calculated from
[volumetric CSD profiles](#data-generated-csd_profiles-profiles_npz)
with FEM (_FEniCS_).  For that purpose
[mesh](#data-generated-meshes),
[model properties](#data-generated-fenics_leadfield_corrections-conductivity_ini),
and element degree (defaults to $1$) and type (defaults to continuous Galerkin)
are specified.

There are two tools:
- `forward_slice_on_plate.py` (for slice model) which additionally
  allows to set the ground potential at a slice-covering dome
  (if not given $0 V$ potential is assumed at infinity),
- `forward_sphere_on_plate.py` (for spherical model) which additionally
  allows to set the Z coordinate of the edge of grounded plate
  (defaults to $-88 mm$).


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


# TODOs

- links from files to tools
