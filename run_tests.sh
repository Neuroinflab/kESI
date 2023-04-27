# !!! IMPORTANT !!!
# run with `source`
for minor_version in {7..10}
do
  python_version=3.${minor_version}
  name=kesi${python_version}
  conda activate $name
  python -m unittest discover
  conda deactivate
done

