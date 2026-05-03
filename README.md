# aidotgrids

## Overview

1. [Quick start](#1-quick-start)
2. [Datasets](#2-datasets)
3. [Contributing](#3-contributing)


## 1. Quick start

Install stable version (option A):

```bash
pip install -U aidotgrids
```

Or, install bleeding-edge version (Option B):

```bash
pip uninstall -y aidotgrids         # ensure the PyPI version is removed
pip install git+https://github.com/AI-grids/aidotgrids
```

## 2. Datasets

All standardized tasks are hosted on Hugging Face Hub and can be downloaded 
automatically via `aidotgrids.load`. Current coverage:

| Task / Dataset          | Modality                    | Docs                                   |
| ----------------------- | --------------------------- | -------------------------------------- |
| **OPFData**             | Graphs (optimal-power-flow) | [details](docs/opfdata.md)             |
| **PowerGraph**          | Transmission-grid graphs    | [details](docs/powergraph.md)          |
| **SolarCube**           | Satellite imagery           | [details](docs/solarcube.md)           |
| **BuildingElectricity** | Time-series load profiles   | [details](docs/buildingelectricity.md) |
| **WindFarm**            | SCADA & weather             | [details](docs/windfarm.md)            |


Load a dataset in a few lines:

```Python
from aidotgrids import load

ds = load(
    task_name="OPFData",
    subtask_name="train_small_test_medium",
    root_path="~/AI-grids"  # local cache directory
)
```

## 3. Contributing

We welcome pull requests for new datasets, models and algorithms, or simply a 
fix in our code. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening 
an issue or pull request.