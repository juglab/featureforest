It is highly recomended to create and use a new environment to install the plugin and its dependencies.
You can use [mamba] or [conda] to manage you environments but it's not necessary and you can use python `venv` as well. 

## Setup
We provided `install.sh` for Linux & Mac OS users, and `install.bat` for Windows users.  
First you need to clone the repo:
```bash
git clone https://github.com/juglab/featureforest
cd ./featureforest
```
Now run the installation script:
```bash
# Linux or Mac OS
sh ./install.sh
```
```bash
# Windows
./install.bat
```

## Step by Step
1. Create a new environment:
```bash
conda create -n featureforest -y python=3.10
```

2. Activate the environment:
```bash
conda activate featureforest
```

3. Install `torch` and `torchvision`:  
You can follow the instruction from [here](https://pytorch.org/get-started/locally/). But we can also use [`light-the-torch`](https://github.com/Slicer/light-the-torch) package:
```bash
pip install light-the-torch
ltt install 'torch>=2.5.1' 'torchvision>=0.20.1'
```
This will install the appropriate PyTorch binaries without user intervention by automatically identifying compatible CUDA versions from the local setup.  

4. Installing all other dependencies:
```bash
pip install -r ./requirements.txt
```
This will install all dependencies including `napari`, `segment-anything` and `sam-2`.  

5. Finally, install the plugin:
```bash
pip install git+https://github.com/juglab/featureforest.git
```

## Requirements
- `python>=3.10`
- `numpy<2.2`
- `pytorch>=2.5.1`
- `torchvision>=0.20.1`
- `timm`
- `segment-anything`
- `sam-2`
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `scipy`
- `matplotlib`
- `pyqt`
- `magicgui`
- `qtpy`
- `napari`
- `h5py`
- `pynrrd`
- `pooch`


There is also a [pypi package](https://pypi.org/project/featureforest/) available that you can install **FF** using `pip`:
```bash
pip install featureforest
```
!!! note
    Before install `featureforest` using `pip` you need to install `segment-anything` and `sam-2` manually.

For detailed napari installation see [here](https://napari.org/stable/tutorials/fundamentals/installation).  


[conda]: https://conda.io/projects/conda/en/latest/index.html
[mamba]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
