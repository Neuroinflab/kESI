#!/bin/bash
# USAGE:
# $ source jupyter_server.sh
#
# REQUIRES:
# $ openssl req -x509 -nodes -days 365 -newkey rsa:4096 -keyout .jupyter_nb.key -out .jupyter_nb.pem
# $ conda create -n jupyter --no-default-packages ipython jupyter nbpresent nb_conda nb_conda_kernels
conda activate jupyter
jupyter notebook --debug --no-browser --certfile .jupyter_nb.pem --keyfile .jupyter_nb.key --ip $(hostname) --port 8888
