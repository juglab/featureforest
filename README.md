# Feature Forest

[![License BSD-3](https://img.shields.io/pypi/l/featureforest.svg?color=green)](https://github.com/juglab/featureforest/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/featureforest.svg?color=green)](https://pypi.org/project/featureforest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/featureforest)
[![Python Version](https://img.shields.io/pypi/pyversions/featureforest.svg?color=green)](https://python.org)
[![tests](https://github.com/juglab/featureforest/workflows/tests/badge.svg)](https://github.com/juglab/featureforest/actions)
[![codecov](https://codecov.io/gh/juglab/featureforest/branch/main/graph/badge.svg)](https://codecov.io/gh/juglab/featureforest)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/featureforest)](https://napari-hub.org/plugins/featureforest)
<!--[![Downloads](https://pepy.tech/badge/featureforest)](https://pepy.tech/project/featureforest)-->

**A napari plugin for making image annotation using feature space of vision transformers and random forest classifier.**
We developed a *napari* plugin to train a *Random Forest* model using extracted features of vision foundation models and just a few scribble labels provided by the user as input. This approach can do the segmentation of desired objects almost as well as manual segmentations but in a much shorter time with less manual effort.

----------------------------------

## Documentation
You can check the documentation [here](https://juglab.github.io/featureforest/) (⚠️ work in progress!).

## Installation
To install this plugin you need to use [conda] or [mamba] to create an environment and install the requirements. Use commands below to create the environment and install the plugin:
```bash
git clone https://github.com/juglab/featureforest
cd ./featureforest
```
```bash
# for GPU
conda env create -f ./env_gpu.yml
```
```bash
# if you don't have a GPU
conda env create -f ./env_cpu.yml
```

For developers that want to contribute to FeatureForest, you need to use this command to install the `dev` dependencies:
```bash
pip install -U "featureforest[dev]"
```
And make sure you have `pre-commit` installed in your environment, before committing changes:
```bash
pre-commit install
```

For more detailed installation guide, check out [here](https://juglab.github.io/featureforest/install/).


## Cite us

Seifi, Mehdi, Damian Dalle Nogare, Juan Battagliotti, Vera Galinova, Ananya Kediga Rao, AI4Life Horizon Europe Programme Consortium, Johan Decelle, Florian Jug, and Joran Deschamps. "FeatureForest: the power of foundation models, the usability of random forests." bioRxiv (2024): 2024-12. [DOI: 10.1101/2024.12.12.628025](https://www.biorxiv.org/content/10.1101/2024.12.12.628025v1.full)


## License

Distributed under the terms of the [BSD-3] license,
"featureforest" is free and open source software

## Issues

If you encounter any problems, please [file an issue](https://github.com/juglab/featureforest/issues/new) along with a detailed description.

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
