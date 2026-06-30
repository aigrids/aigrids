# aigrids

## Overview

1. [Quick start](#1-quick-start)
2. [Datasets](#2-datasets)
3. [Contributing](#3-contributing)


## 1. Quick start

Install stable version (option A):

```bash
pip install -U aigrids
```

Or, install bleeding-edge version (Option B):

```bash
pip uninstall -y aigrids         # ensure the PyPI version is removed
pip install git+https://github.com/aigrids/aigrids
```

## 2. Datasets

All standardized tasks are hosted on Hugging Face Hub and can be downloaded 
automatically via `aigrids.load`. Current coverage:

| Task / Dataset          | Modality                    | Docs                                   |
| ----------------------- | --------------------------- | -------------------------------------- |
| **OPFData**             | Graphs (optimal-power-flow) | [details](docs/opfdata.md)             |
| **PowerGraph**          | Transmission-grid graphs    | [details](docs/powergraph.md)          |
| **SolarCube**           | Satellite imagery           | [details](docs/solarcube.md)           |
| **BuildingElectricity** | Time-series load profiles   | [details](docs/buildingelectricity.md) |
| **WindFarm**            | SCADA & weather             | [details](docs/windfarm.md)            |


Load a dataset in a few lines:

```Python
from aigrids import load

ds = load.load_task(
    task_name="WindFarm",
    subtask_name="odd_time_predict48h",
    root_path="~/AI-grids"  # local cache/download directory
)
```

## 3. Contributing

We welcome pull requests for new datasets, models and algorithms, or simply a 
fix in our code. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening 
an issue or pull request.