# !!! IMPORTANT !!!
# run with `source`
# 3.5 -> no gcc package required by fenics
conda install -c conda-forge --yes mamba
for python_version in "3.6" "3.7" "3.8"
do
  name=kesi${python_version/./}
  mamba create --name $name --no-default-packages --yes -c conda-forge -c bioconda snakemake python=$python_version
  conda activate $name
  mamba install -c conda-forge --yes numpy scipy matplotlib pandas ipython ipykernel importlib_metadata
  mamba install -c conda-forge --yes fenics meshio
  python setup.py develop
  conda deactivate
done

