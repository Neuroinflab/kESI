# !!! IMPORTANT !!!
# run with `source`
# 3.5 -> no gcc package required by fenics
# 3.6 -> no pyproject.toml support
for minor_version in {7..10}
do
  python_version=3.${minor_version}
  name=kesi${python_version}
  conda create --name $name --no-default-packages --yes -c conda-forge mamba python=$python_version
  conda activate $name
  mamba install -c conda-forge -c bioconda --yes snakemake
  mamba install -c conda-forge --yes numpy scipy matplotlib pandas ipython ipykernel importlib_metadata
## there is at least one system in which ipython_genutils are missing if not installed explicitely
#  mamba install -c conda-forge --yes numpy scipy matplotlib pandas ipython ipython_genutils ipykernel importlib_metadata
  mamba install -c conda-forge --yes fenics meshio
  pip install -e .
  mamba clean -a --yes
  conda deactivate
done

