# !!! IMPORTANT !!!
# run with `source`
for python_version in "3.7" "3.8" "3.9" "3.10"
do
  name=kesi${python_version}
  conda activate $name
  python -m unittest discover
  conda deactivate
done

