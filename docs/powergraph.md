# PowerGraph

## Load standardized task data

You can both download and load all data associated with standardized PowerGraph tasks
in the following fashion. For example, load the train_small_test_medium sub-task
with:

```Python
from aigrids import load_task

dataset = load_task(
    task_name='PowerGraph', 
    subtask_name='cascading_failure_binary',
    root_path='~/AI-grids/'
)
```

Available sub-tasks are:
- `cascading_failure_binary`
- `cascading_failure_multiclass`
- `demand_not_served`
- `cascading_failure_sequence`


## Download raw task data

You can download the raw PowerGraph datasets from 
https://figshare.com/articles/dataset/PowerGraph/22820534. The download can be
done programmatically, and files be decompressed, using the following command

```bash
wget https://figshare.com/ndownloader/articles/22820534/versions/5
mv 5 dataset.zip
unzip dataset.zip
```

