# OPFData

## Load standardized task data

You can both download and load all data associated with standardized OPFData task
in the following fashion. For example, load the train_small_test_medium sub-task
with:

```Python
from aigrids import load_task

dataset = load_task(
    task_name='OPFData', 
    subtask_name='train_small_test_medium',
    root_path='~/AI-grids/'
)
```

Available sub-tasks are:
- `train_small_test_medium`
- `train_small_test_large`
- `train_medium_test_small`
- `train_medium_test_large`
- `train_large_test_small`
- `train_large_test_medium`


## Download raw task data

You can download the raw OPFData datasets using Pytorch Geometric, or a Google 
Cloud bucket under
https://console.cloud.google.com/storage/browser/gridopt-dataset/ where the data 
is stored.


```python
!pip install pyg-nightly # Only necessary until PyG 2.6.0 is released.

from torch_geometric.datasets import OPFDataset

# set directory in which you want to download and extract dataset
root = "YOUR_ROOT_DIRECTORY"

# set name of grid topology dataset
case_name = "GRID_TOPOLOGY_NAME" # e.g. pglib_opf_case14_ieee

# choose if you want dataset with n-1 topological perturbations or not.
perturbation = True # set True or False

OPFDataset(
    root=root,
    case_name=case_name,
    topological_perturbations=perturbation
)
```

The full list of available case names is:
- `pglib_opf_case14_ieee`
- `pglib_opf_case30_ieee`
- `pglib_opf_case57_ieee`
- `pglib_opf_case118_ieee`
- `pglib_opf_case500_goc`
- `pglib_opf_case2000_goc`
- `pglib_opf_case4661_sdet`
- `pglib_opf_case6470_rte`
- `pglib_opf_case10000_goc`
- `pglib_opf_case13659_pegase`

Each of these cases can be downloaded with perturbation once set to True and 
once set to False.
