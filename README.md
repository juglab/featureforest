# Feature Forest

[![License BSD-3](https://img.shields.io/pypi/l/featureforest.svg?color=green)](https://github.com/juglab/featureforest/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/featureforest.svg?color=green)](https://pypi.org/project/featureforest)
[![Python Version](https://img.shields.io/pypi/pyversions/featureforest.svg?color=green)](https://python.org)
[![tests](https://github.com/juglab/featureforest/workflows/tests/badge.svg)](https://github.com/juglab/featureforest/actions)
[![codecov](https://codecov.io/gh/juglab/featureforest/branch/main/graph/badge.svg)](https://codecov.io/gh/juglab/featureforest)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/featureforest)](https://napari-hub.org/plugins/featureforest)

**A napari plugin for making image annotation using feature space of vision transformers and random forest classifier.**  
We developed a *napari* plugin to train a *Random Forest* model using extracted features of vision foundation models and just a few scribble labels provided by the user as input. This approach can do the segmentation of desired objects almost as well as manual segmentations but in a much shorter time with less manual effort.

----------------------------------

## Documentation
The plugin documentation is [here](docs/index.md).

## Installation
To install this plugin you need to use [conda] or [mamba] to create a environment and install the requirements. Use the commands below to create the environment and install the plugin:
```bash
# for GPU
conda env create -f ./env_gpu.yml
```
```bash
# if you don't have a GPU
conda env create -f ./env_cpu.yml
```

#### Note: You need to install `sam-2` which can be install easily using conda. To install `sam-2` using `pip` please refer to the official [sam-2](https://github.com/facebookresearch/sam2) repository.

### Requirements
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

If you want to install the plugin manually using GPU, please follow the pytorch installation instruction [here](https://pytorch.org/get-started/locally/).  
For detailed napari installation see [here](https://napari.org/stable/tutorials/fundamentals/installation).  

### Installing The Plugin
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


## License

Distributed under the terms of the [BSD-3] license,
"featureforest" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[conda]: https://conda.io/projects/conda/en/latest/index.html
[mamba]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html