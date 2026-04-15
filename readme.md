# kESI

Kernel based curent source density estimation using FEM models of conductivity.

## Basic structure

`src/kesi` - contains the kESI framework

`extras` - contains tutorials and snakemake file which is used to do everything - from mesh segmentation to FEM elements to leadfield and CSD calculations.

# Installation

Create and activate python environment with python 3.9 or higher (tested 3.10) repository.
Afterwards use pip to install the kESI package:

`pip install .`

it should install kESI and required dependendies. By default FEM computation part of the framework is not installed.
As it is not required to compute CSD solutions using kCSD method or when you already have inverse leadfields precomputed and can be cumbersome to install on some PC configurations.
To install kESI with FEM calculation dependencies (tested under Linux enviroments) run:

`pip install .[fem]`

To ensure you are using verified working dependency versions, you can use UV (https://github.com/astral-sh/uv) to install the tested dependencies locked in a lock file. Simply run the `uv sync --locked` command, which will create a virtual environment using appropirate Python version with all the dependencies locked in a lock file and install kESI in that environment.

Additionally, kESI requires `gmsh` for mesh generation and FEM segmentation. You'll have to install it from your package manager for example Ubuntu:

`sudo apt install gmsh` or `pip install gmsh`

