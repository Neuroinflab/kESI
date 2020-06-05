# !!! IMPORTANT !!!
# run with `source`
# 3.5 -> meshio/h5py conflicted with numpy
# 3.7+ -> snakemake unavailable
# for python_version in "3.6"
for python_version in "3.6"
do
  name=kesi${python_version/./}
  conda create --name $name --no-default-packages --yes numpy scipy matplotlib pandas ipython ipykernel importlib_metadata python=$python_version
  conda activate $name
  conda install -c bioconda --yes snakemake
  conda install -c conda-forge --yes fenics meshio
  python setup.py develop
  conda deactivate
done

