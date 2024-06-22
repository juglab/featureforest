# Feature Forest

[![License BSD-3](https://img.shields.io/pypi/l/napari-sam-labeling-tools.svg?color=green)](https://github.com/juglab/napari-sam-labeling-tools/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sam-labeling-tools.svg?color=green)](https://pypi.org/project/napari-sam-labeling-tools)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sam-labeling-tools.svg?color=green)](https://python.org)
[![tests](https://github.com/juglab/napari-sam-labeling-tools/workflows/tests/badge.svg)](https://github.com/juglab/napari-sam-labeling-tools/actions)
[![codecov](https://codecov.io/gh/juglab/napari-sam-labeling-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/juglab/napari-sam-labeling-tools)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sam-labeling-tools)](https://napari-hub.org/plugins/napari-sam-labeling-tools)

A napari plugin for segmentation using vision transformers' features.  
We developed a *napari* plugin to train a *Random Forest* model using extracted embeddings of ViT models for input and just a few scribble labels provided by the user. This approach can do the segmentation of desired objects almost as well as manual segmentations but in a much shorter time with less manual effort.

----------------------------------

## Documentation
The plugin documentation is [here](docs/index.md).

## Installation
It is highly recommended to use a python environment manager like [conda] to create a clean environment for installation.  
You can install all the requirements using provided environment config file ([`env.yml`](env.yml)):  
```bash
conda env create -f ./env.yml
```

### Requirements
- `python >= 3.9`
- `numpy`
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `pyqt`
- `magicgui`
- `qtpy`
- `napari`
- `h5py`
- `pytorch`
- `torchvision`
- `timm`
- `pynrrd`

If you want to use GPU, please follow the pytorch installation instruction [here](https://pytorch.org/get-started/locally/).  
For detailed napari installation see [here](https://napari.org/stable/tutorials/fundamentals/installation).  

### Installing The Plugin
If you use the conda `env.yml` file, the plugin will be installed automatically. But in case you already have the environment setup, 
you can just install the plugin. First clone the repository:
```bash
git clone https://github.com/juglab/featureforest
```
Then run the following commands:
```bash
cd ./featureforest
pip install .
```

<!-- You can install `napari-sam-labeling-tools` via [pip]:

    pip install napari-sam-labeling-tools -->




<!-- ## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request. -->

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
