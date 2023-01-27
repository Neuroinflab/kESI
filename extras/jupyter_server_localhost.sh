#!/bin/bash
# USAGE:
# $ source jupyter_server_localhost.sh
#
# REQUIRES:
# $ conda create -n jupyter --no-default-packages ipython jupyter
# $ conda activate jupyter
# $ conda install nb_conda nb_conda_kernels
# $ conda deactivate
conda activate jupyter
jupyter notebook --debug
