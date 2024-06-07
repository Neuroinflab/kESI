# kESI

Kernel based curent source density estimation using FEM models of conductivity.

## Basic structure

`src/kesi` - contains a very barebones kESI framewo
rk

`extras` - contains tutorials and snakemake file which is used to do everything - from mesh segmentation to FEM elements to leadfield and CSD calculations. Refer to `SNAKEMAKE_README.md`.

# Installation

You need to use conda to install the appropriate dependencies defined in `enviroment_*.yml` files, `*` stands for different Python versions.
To install kESI with the legacy FEniCS for Python 3.10 you are required to use conda:

`conda env create -f enviroment_3.10.yml`

Activate `kesi_310` enviroment:

`conda activate kesi_310` 

afterwards use pip to install the kESI package:

`pip install .`

If you wish to only use modern MFEM solver, you can install using pip without any additional package manager requirement.

Additionally, kESI requires `gmsh` for mesh generation and FEM segmentation. You'll have to install it from your package manager for example Ubuntu:

`sudo apt install gmsh` or `pip install gmsh`

Kesi will be available in this conda enviroment.

