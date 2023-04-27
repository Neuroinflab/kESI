# !!! IMPORTANT !!!
# run with `source`
# 3.5 -> no gcc package required by fenics
# 3.6 -> no pyproject.toml support
for python_version in "3.7" "3.8" "3.9" "3.10"
do
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

