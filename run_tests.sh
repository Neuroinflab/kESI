# !!! IMPORTANT !!!
# run with `source`
# for python_version in "3.6"
for python_version in "3.6"
do
  name=kesi${python_version/./}
  conda activate $name
  python -m unittest discover
  conda deactivate
done

