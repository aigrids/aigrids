# WindFarm

## Load standardized task data

You can both download and load all data associated with standardized WindFarm
tasks in the following fashion. For example, load the `odd_time_predict48h` 
sub-task with:

```Python
from aigrids import load_task

dataset = load_task(
    task_name='WindFarm', 
    subtask_name='odd_time_predict48h',
    root_path='~/AI-grids/'
)
```

Available sub-tasks are:
- `odd_time_predict48h`
- `odd_space_predict48h`
- `odd_spacetime_predict48h`
- `odd_time_predict72h`
- `odd_space_predict72h`
- `odd_spacetime_predict72h`


## Download raw task data

You can download the raw WindFarm datasets from FigShare at 
https://doi.org/10.6084/m9.figshare.24798654.

Programmatically, we download the dataset with:

```bash
wget https://figshare.com/ndownloader/articles/24798654/versions/2
mv 2 WindFarm.zip
unzip WindFarm
rm WindFarm.zip
mv SDWPF_dataset WindFarm
```

