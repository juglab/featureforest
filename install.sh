#! /bin/sh
# This script will install FeatureForest and its dependencies in a new conda environment.
echo "\nCreate a new conda environment named featureforest with Python 3.10"
conda init
conda create -n featureforest -y python=3.10
echo "\nActivate the featureforest environment"
source activate base
conda activate featureforest
echo "\nInstall PyTorch and torchvision using light-the-torch"
python -m pip install light-the-torch
ltt install 'torch>=2.5.1' 'torchvision>=0.20.1'
echo "\nInstall additional dependencies"
python -m pip install -r ./requirements.txt
echo "\nInstall FeatureForest from GitHub"
python -m pip install git+https://github.com/juglab/featureforest.git
