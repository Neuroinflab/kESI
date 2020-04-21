#!/bin/bash
# USAGE:
# $ source jupyter_server.sh
#
# REQUIRES:
# $ openssl req -x509 -nodes -days 365 -newkey rsa:4096 -keyout .jupyter_nb.key -out .jupyter_nb.pem
# $ conda create -n kesi37 python==3.7 ipython jupyter matplotlib nbpresent nb_conda nb_conda_kernels numpy pandas scipy
conda activate kesi37
jupyter notebook --no-browser --certfile .jupyter_nb.pem --keyfile .jupyter_nb.key --ip $(hostname) --port 8888
