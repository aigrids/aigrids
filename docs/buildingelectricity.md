# BuildingElectricity

## Load standardized task data

You can both download and load all data associated with standardized BuildingElectricity tasks
in the following fashion. For example, load the `buildings_451` sub-task
with:

```Python
from aigrids import load_task

dataset = load_task(
    task_name='BuildingElectricity', 
    subtask_name='odd_time_buildings92',
    root_path='~/AI-grids/'
)
```

Available sub-tasks are:
- `odd_time_buildings92`
- `odd_space_buildings92`
- `odd_spacetime_buildings92`
- `odd_time_wload_buildings92`
- `odd_space_wload_buildings92`
- `odd_spacetime_wload_buildings92`
- `odd_time_buildings451`
- `odd_space_buildings451`
- `odd_spacetime_buildings451`
- `odd_time_wload_buildings451`
- `odd_spac_wloade_buildings451`
- `odd_spacetime_wload_buildings451`

## Download raw task data

You can download the raw BuildingElectricity datasets from Harvard Dataverse at
 https://doi.org/10.7910/DVN/3VYYET.