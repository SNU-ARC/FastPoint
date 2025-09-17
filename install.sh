#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

# install PyTorch
conda deactivate
conda env remove --name fastpoint
conda create -n fastpoint -y python=3.7 numpy=1.20 numba
conda activate fastpoint

conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia

# install relevant packages
# torch-scatter is a must, and others are optional
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
# pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r requirements.txt
pip install shortuuid

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

cd chamfer_dist
python setup.py install --user
