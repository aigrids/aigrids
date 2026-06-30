"""Loads requested subtask for PowerGraph.

"""
import os
import gc
import json
import random
import logging
from typing import Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ALL_GRIDS_LIST = [
    'ieee24', 
    'ieee39', 
    'ieee118', 
    'uk'
]

LIST_AVAIL_SUBTASKNAMES = [
    'cascading_failure_binary',
    'cascading_failure_multiclass',
    'demand_not_served_regression',
    'cascading_failure_sequence'
]

# train, val, test
SPLIT_RATIO = (0.5, 0.1, 0.4)  

# Set default dtypes
np_dtype = np.float64


def load(
    local_dir: str,
    subtask_name: str,
    data_frac: Union[int, float],
    train_frac: Union[int, float],
    max_workers: int,
    seed: int = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load and prepare the data for a given subtask.

    """
    # check if valid subtask name is passed
    if subtask_name not in LIST_AVAIL_SUBTASKNAMES:
        raise ValueError(f"Unknown subtask name: {subtask_name}")
        
    # Load JSON files
    data_dict = _load_json_files(local_dir, data_frac, max_workers=max_workers)

    # Parse into subtask datasets
    data_list = _parse_dataset(data_dict, subtask_name)

    # Shuffle data
    random.Random(seed).shuffle(data_list)

    # Split into training, validation, and testing
    train_data, val_data, test_data = _split_dataset(data_list, train_frac)

    # Clean up large dictionary to free memory
    del data_dict
    gc.collect()

    # Load natural language descriptions
    task_description    = _create_taskdescription()
    subtask_description = _create_subtaskdescription(subtask_name)

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'task_description': task_description,
        'subtask_description': subtask_description
    }


def _parse_dataset(
    data_dict: Dict[str, Any], 
    subtask_name: str
) -> List[Dict[str, Any]]:
    """
    Parse data dictionary by subtask name, extracting and transforming nodes,
    edges, edge_index, and labels as required. Returns a list of parsed samples.
    """

    # Map each subtask_name to the needed label key
    subtask_label_map = {
        'cascading_failure_binary': '1',
        'cascading_failure_multiclass': '2',
        'demand_not_served_regression': '3',
        'cascading_failure_sequence': '4'
    }

    label_key = subtask_label_map[subtask_name]
    for entry in data_dict.values():
        entry['labels'] = entry['labels'][label_key]

    data_list = []
    for _ in range(len(data_dict)):
        # popitem removes an arbitrary (key, value) pair
        _, entry = data_dict.popitem()

        # --- Parse nodes ---
        pnet = entry['nodes']['1']
        snet = entry['nodes']['2']
        v = entry['nodes']['3']
        x_node = np.stack([pnet, snet, v], axis=1)

        # --- Parse edges ---
        # Note: 1-4 are from Ef_nc, 5-8 are from Ef; here we use 5-8.
        pij = entry['edges']['5']
        qij = entry['edges']['6']
        xij = entry['edges']['7']
        lrij = entry['edges']['8']
        x_edge = np.stack([pij, qij, xij, lrij], axis=1)

        # --- Parse edge index ---
        from_node = entry['edge_index']['1']
        to_node   = entry['edge_index']['2']
        edge_index = np.stack([from_node, to_node], axis=1)

        labels = entry['labels']
        data_list.append(
            {
                'x_node': x_node,
                'x_edge': x_edge,
                'edge_index': edge_index,
                'labels': labels
            }
        )

    return data_list


def _backupfunction(
    data_dict: Dict[str, Any], 
    subtask_name: str
) -> Dict[str, Any]:
    """
    Backup function for parsing data in-place. Not used in the primary workflow.
    """
    for entry in data_dict.values():
        # --- Parse nodes ---
        pnet = entry['nodes']['1']
        snet = entry['nodes']['2']
        v    = entry['nodes']['3']
        del entry['nodes']
        entry['x_node'] = np.stack([pnet, snet, v], axis=1)

        # --- Parse edges ---
        pij  = entry['edges']['5']
        qij  = entry['edges']['6']
        xij  = entry['edges']['7']
        lrij = entry['edges']['8']
        del entry['edges']
        entry['x_edge'] = np.stack([pij, qij, xij, lrij], axis=1)

        # --- Parse edge index ---
        from_node = entry['edge_index']['1']
        to_node   = entry['edge_index']['2']
        entry['edge_index'] = np.stack([from_node, to_node], axis=1)

        # Parse labels similarly (same logic as in _parse_dataset).
        if subtask_name == 'cascading_failure_binary':
            entry['labels'] = entry['labels']['1']
        elif subtask_name == 'cascading_failure_multiclass':
            entry['labels'] = entry['labels']['2']
        elif subtask_name == 'demand_not_served_regression':
            entry['labels'] = entry['labels']['3']
        elif subtask_name == 'cascading_failure_sequence':
            entry['labels'] = entry['labels']['4']

    return data_dict


def _split_dataset(
    data_list: List[Dict[str, Any]],
    train_frac: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split the dataset into train, validation, and test sets, based on SPLIT_RATIO.
    Then within the train-slice, only a fraction (train_frac) is used for actual
    training, discarding the remainder of the 'train slice'.
    """
    total_size = len(data_list)
    size_train = int(total_size * SPLIT_RATIO[0])
    size_val   = int(total_size * SPLIT_RATIO[1])

    id_end_train = int(size_train * train_frac)
    id_start_val = size_train
    id_end_val   = size_train + size_val

    # Slice the data
    train_data = data_list[:id_end_train]
    val_data = data_list[id_start_val:id_end_val]
    test_data = data_list[id_end_val:]

    return train_data, val_data, test_data


def _load_json_files(
    local_dir: str,
    data_frac: Union[int, float],
    max_workers: int = 1
) -> Dict[str, Any]:
    """
    Load JSON files from each grid in ALL_GRIDS_LIST, accumulating them into a 
    single dictionary.
    """
    combined_data_dict = {}

    for gridname in ALL_GRIDS_LIST:
        path_grid = os.path.join(local_dir, gridname)
        if not os.path.isdir(path_grid):
            logging.warning(f"Directory '{path_grid}' does not exist, skipped.")
            continue

        file_list_grid = [f for f in os.listdir(path_grid) if f.endswith('.json')]
        total_files = len(file_list_grid)
        if total_files == 0:
            logging.info(f"No JSON files found in '{path_grid}', skipped.")
            continue

        # Determine how many files to load based on data_frac
        num_to_load = int(round(data_frac * total_files))
        file_list_grid = file_list_grid[:num_to_load]
        logging.info(f"Loading {num_to_load} files from '{path_grid}'.")

        def _read_json_file(filename: str) -> Dict[str, Any]:
            with open(filename, 'r', encoding='utf-8') as fh:
                return json.load(fh)

        partial_dicts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_read_json_file, os.path.join(path_grid, fname)): fname
                for fname in file_list_grid
            }
            for future in as_completed(future_map):
                fname = future_map[future]
                try:
                    file_data = future.result()
                    partial_dicts.append(file_data)
                except Exception as e:
                    logging.error(f"Error reading '{fname}': {e}")

        # Merge each file's dictionary into the global data structure
        for partial_dict in partial_dicts:
            combined_data_dict.update(partial_dict)

        # Clean up
        del partial_dicts
        gc.collect()

    return combined_data_dict


def _create_taskdescription():
    """Contains natural language description of task. Placeholder."""

    task_description = """
    Given the steady state of a power system at a snapshot in time, our goal in 
    PowerGraph is to predict the possibility and resulting characteristics of a 
    cascading failure, which is the outage of a grid component that triggers a 
    chain reaction of subsequent outages.

    Improving the computational speed of this analysis allows power system 
    operators to more frequently assess and revise control actions on their grid, 
    and thereby better prevent failure modes associated with the integration of 
    fluctuating renewable energy sources.

    The main physical law that solutions to this task must satisfy is Kirchhoff's 
    Current Law.

    We define four sub-task datasets, each sharing the same input but differing 
    in the predicted characteristic of a cascading failure, which serves as the 
    label. Specifically, we distinguish:

    cascading_failure_binary: A binary classification task predicting the 
    probability of a cascading failure.

    cascading_failure_multiclass: A multi-class classification task predicting 
    the probability of four distinct combinations of cascading failures and 
    demand-not-served scenarios: demand not served larger than zero and cascading 
    failure occurs, demand not served larger than zero and no cascading failure 
    occurs, demand not served is zero and cascading failure occurs, demand not 
    served is zero and no cascading failure occurs.

    demand_not_served_regression: A regression task predicting the amount of 
    power demand left unserved due to a potential cascading failure.

    cascading_failure_sequence: A sequential prediction task forecasting the 
    order in which transmission lines may fail during a cascading failure.
    """

    return task_description



def _create_subtaskdescription(subtask_name: str):
    """Contains natural language description of subtask. Placeholder."""

    subtask_description = f"""
    Here, we are solving instances of the {subtask_name} subtask.
    """.format(subtask_name)
    
    return subtask_description


