## Easy Way!
To install this plugin you need to use [mamba] or [conda] to create a environment and install the requirements. Use commands below to create the environment and install the plugin:
```bash
git clone https://github.com/juglab/featureforest
cd ./featureforest
```
```bash
# for GPU
mamba env create -f ./env_gpu.yml
```
```bash
# if you don't have a GPU
mamba env create -f ./env_cpu.yml
```

### Note 
You need to install `sam-2` which can be installed easily using mamba (or conda). To install `sam-2` using `pip` please refer to the official [sam-2](https://github.com/facebookresearch/sam2) repository.

## Requirements
- `python >= 3.10`
- `numpy==1.24.4`
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `pyqt`
- `magicgui`
- `qtpy`
- `napari`
- `h5py`
- `pytorch=2.3.1`
- `torchvision=0.18.1`
- `timm=1.0.9`
- `pynrrd`
- `segment-anything`
- `sam-2`

## Installing Only The Plugin
If you use the provided conda environment yaml files, the plugin will be installed automatically. But in case you already have the environment setup, 
you can just install the plugin. First clone the repository:
```bash
git clone https://github.com/juglab/featureforest
```
Then run the following commands:
```bash
cd ./featureforest
pip install .
```

There is also a [pypi package](https://pypi.org/project/featureforest/) available that you can install using `pip`:
```bash
pip install featureforest
```

If you want to install the plugin manually using GPU, please follow the pytorch installation instruction [here](https://pytorch.org/get-started/locally/). For detailed napari installation see [here](https://napari.org/stable/tutorials/fundamentals/installation).  


[conda]: https://conda.io/projects/conda/en/latest/index.html
[mamba]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html