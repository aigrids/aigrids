# SolarCube

## Load standardized task data

You can both download and load all data associated with standardized SolarCube tasks
in the following fashion. For example, load the `odd_time_area_3h` sub-task
with:

```Python
from aigrids import load_task

dataset = load_task(
    task_name='SolarCube', 
    subtask_name='odd_time_area_3h',
    root_path='~/AI-grids/'
)
```

Available sub-tasks are:
- `odd_time_area`
- `odd_time_point`
- `odd_space_area`
- `odd_space_point`
- `odd_spacetime_area`
- `odd_spacetime_point`



## Download raw task data

You can download the raw SolarCube datasets from 
https://zenodo.org/records/11498739. The download can be
done programmatically, and files be decompressed, using the following command

```bash
wget "https://zenodo.org/api/records/11498739/files-archive"
mv files-archive dataset.zip
unzip dataset.zip
```