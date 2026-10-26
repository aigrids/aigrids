# SystemImbalance

## Load standardized task data

You can both download and load all data associated with standardized 
SystemImbalance tasks in the following fashion. For example, load the 
`odd_time_volume_predict1h` sub-task with:

```Python
from aigrids import load_task

dataset = load_task(
    task_name='SystemImbalance', 
    subtask_name='odd_time_volume_predict1h',
    root_path='~/AI-grids/'
)
```

Available sub-tasks are:
- `odd_time_volume_predict1h`
- `odd_space_volume_predict1h`
- `odd_spacetime_volume_predict1h`
- `odd_time_price_predict1h`


## Download raw task data

All data is published by Bundesnetzagentur | SMARD.de under a CC BY 4.0 
licence. The raw series as downloaded from SMARD in October 2026 are provided 
at https://huggingface.co/datasets/nikakhosravi-acs/SystemImbalance_raw. The 
download can be done programmatically, and files be decompressed, using the 
following command

```bash
wget https://huggingface.co/datasets/nikakhosravi-acs/SystemImbalance_raw/resolve/main/raw.tar.gz
tar -xzf raw.tar.gz
```

We merge the raw series into the standardized data with the processing script 
of our data collection repository, after setting `path_root_source` and 
`path_root_target` in the script:

```bash
git clone https://github.com/aigrids/data-collectionv1
python data-collectionv1/processing/systemimbalance_process.py
```
