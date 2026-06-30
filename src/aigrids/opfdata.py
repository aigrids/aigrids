"""Loads requested subtask for OPFData.

"""
import os
import gc
import math
import json
import random
import numpy as np
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import dataset_utils

SMALL_GRIDS_LIST = [
    'pglib_opf_case14_ieee',
    'pglib_opf_case30_ieee',
    'pglib_opf_case57_ieee'
]

MEDIUM_GRIDS_LIST = [
    'pglib_opf_case118_ieee',
    'pglib_opf_case500_goc',
    'pglib_opf_case2000_goc'
]

LARGE_GRIDS_LIST = [
    'pglib_opf_case4661_sdet',
    'pglib_opf_case6470_rte',
    'pglib_opf_case10000_goc',
    'pglib_opf_case13659_pegase'
]

LIST_AVAIL_SUBTASKNAMES = [
    'train_small_test_medium',
    'train_small_test_large',
    'train_medium_test_small',
    'train_medium_test_large',
    'train_large_test_small',
    'train_large_test_medium',
]

# train, val, test
SPLIT_RATIO = (0.5, 0.1, 0.4)

# Set default dtypes
np_dtype = np.float64


def load(
    local_dir: str, 
    subtask_name: str,
    data_frac: float,
    train_frac: float,
    max_workers: int,
    seed: int = None
):
    """
    Load out-of-distribution benchmark by merging grids in one pass.

    """
    # check if valid subtask name is passed
    if subtask_name not in LIST_AVAIL_SUBTASKNAMES:
        raise ValueError(f"Unknown subtask_name: {subtask_name}")

    # 1) Determine which grids to load
    if subtask_name.startswith('train_small'):
        train_grids = SMALL_GRIDS_LIST
    elif subtask_name.startswith('train_medium'):
        train_grids = MEDIUM_GRIDS_LIST
    elif subtask_name.startswith('train_large'):
        train_grids = LARGE_GRIDS_LIST

    if subtask_name.endswith('test_small'):
        test_grids = SMALL_GRIDS_LIST
    elif subtask_name.endswith('test_medium'):
        test_grids = MEDIUM_GRIDS_LIST
    elif subtask_name.endswith('test_large'):
        test_grids = LARGE_GRIDS_LIST


    # 2) Load train/val
    train_val_dataset = _load_multiple_grids(local_dir, train_grids, data_frac, 
        max_workers, seed)
    train_val_dataset = dataset_utils.shuffle_datadict(train_val_dataset, seed)

    # Determine split sizes
    total_size = len(train_val_dataset)
    split_normalize = SPLIT_RATIO[0] + SPLIT_RATIO[1]
    train_ratio = SPLIT_RATIO[0] / split_normalize
    val_ratio = SPLIT_RATIO[1] / split_normalize
    size_train = int(total_size * train_ratio)
    size_val = int(total_size * val_ratio)

    # Determine split IDs
    end_train_id = int(size_train * train_frac)
    start_val_id = size_train
    end_val_id   = size_train + size_val

    # Transform dictionary items to list
    train_val_dataset = list(train_val_dataset.items())

    # Slice list
    train_dataset = dict(train_val_dataset[:end_train_id])
    val_dataset = dict(train_val_dataset[start_val_id:end_val_id])
    del train_val_dataset
    gc.collect()

    # 3) Load test
    test_dataset = _load_multiple_grids(local_dir, test_grids, data_frac, 
        max_workers, seed)
    test_dataset = dataset_utils.shuffle_datadict(test_dataset, seed)

    # Determine split size
    total_size = len(test_dataset)
    size_test = int(total_size * SPLIT_RATIO[2])

    # Transform to list and slice, then transform back to dict
    test_dataset = list(test_dataset.items())[:size_test]
    test_dataset = dict(test_dataset)

    # 4) Parse data
    train_dataset = _parse_data(train_dataset)
    val_dataset   = _parse_data(val_dataset)
    test_dataset  = _parse_data(test_dataset)

    # 5) Problem functions
    loss_functions = {
        'obj_gen_cost': obj_gen_cost,
        'eq_pbalance_re': eq_pbalance_re,
        'eq_pbalance_im': eq_pbalance_im, 
        'ineq_lower_box': ineq_lower_box,
        'ineq_upper_box': ineq_upper_box,
        'eq_difference': eq_difference
    }

    # 6) Load natural language descriptions
    task_description    = _create_taskdescription()
    subtask_description = _create_subtaskdescription(subtask_name)

    # 7) Return task data dictionary
    subtask_data = {
        'train_data': train_dataset,
        'val_data': val_dataset,
        'test_data': test_dataset,
        'loss_functions': loss_functions,
        'task_description': task_description,
        'subtask_description': subtask_description
    }
    return subtask_data


def _parse_data(dataset_dict: dict) -> list:
    """ 
    Iterate over dataset dictionary, parse data points and add to data list
    """
    data_list = []
    for i in range(len(dataset_dict)):
        # popitem removes an arbitrary (key, value) pair
        _, value = dataset_dict.popitem()
        value = _parse_and_aggregate_datapoint(value, i_data=i)
        data_list.append(value)
    return data_list


def _parse_and_aggregate_datapoint(datapoint_dict: dict, i_data: int) -> dict:
    """
    Parse data dictionary into features, return Numpy structures.
    """
    # Some dimension metadata
    baseMVA = datapoint_dict['grid']['context'][0][0][0]
    n = len(datapoint_dict['grid']['nodes']['bus'])
    n_e = (
        len(datapoint_dict['grid']['edges']['ac_line']['features'])
        + len(datapoint_dict['grid']['edges']['transformer']['features'])
    )
    n_g = len(datapoint_dict['grid']['nodes']['generator'])

    load_buses = datapoint_dict['grid']['edges']['load_link']['receivers']
    shunt_buses = datapoint_dict['grid']['edges']['shunt_link']['receivers']
    gen_buses = datapoint_dict['grid']['edges']['generator_link']['receivers']

    # --- Node-level ---
    (Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, basekV_n, vl_n, vu_n, bustype_n) = _set_nodelevel_values(
        n, datapoint_dict, load_buses, shunt_buses
    )

    # --- Gen-level ---
    (
        Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g,
        pg_init_g, qg_init_g, gen_bus_g, Sgl_re_n, Sgl_im_n, Sgu_re_n,
        Sgu_im_n, c0g_n, c1g_n, c2g_n, num_gen_n
    ) = _set_generatorlevel_values(n, n_g, datapoint_dict, gen_buses)

    # --- Edge-level ---
    (
        ij_e, Y_re_e, Y_im_e, Yc_ij_im_e, Yc_ijR_im_e, T_mag_e, T_ang_e, su_e,
        vangl_e, vangu_e, Yc_ij_re_e, Yc_ijR_re_e, Y_re_n, Y_im_n, Y_mag_e, Y_ang_e
    ) = _set_edgelevel_values(n_e, datapoint_dict, Ys_re_n, Ys_im_n)

    # Node feature matrix
    x_node = np.stack(
        [
            Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, 
            Y_re_n, Y_im_n, Sgl_re_n, Sgl_im_n,
            Sgu_re_n, Sgu_im_n, vl_n, vu_n,
            c0g_n, c1g_n, c2g_n, num_gen_n,
            basekV_n, bustype_n
        ],
        axis=1
    )

    # Edge feature matrix
    x_edge = np.stack(
        [
            Y_re_e, Y_im_e, Yc_ij_re_e, Yc_ij_im_e,
            Yc_ijR_re_e, Yc_ijR_im_e, su_e, vangl_e,
            vangu_e, T_mag_e, Y_ang_e
        ],
        axis=1
    )

    # Generator feature matrix
    x_gen = np.stack(
        [
            Sl_re_g, Sl_im_g, Su_re_g, Su_im_g,
            c0_g, c1_g, c2_g, mbase_g, gen_bus_g
        ],
        axis=1
    )

    # Grid feature matrix
    x_grid = np.stack([baseMVA, n, n_e, n_g])

    return {
        "x_node": x_node,
        "x_edge": x_edge,
        "x_gen": x_gen,
        "x_grid": x_grid,
        "edge_index": ij_e,
    }


def _load_multiple_grids(
    local_dir: str, 
    grid_list: list[str], 
    data_frac: float,
    max_workers: int,
    seed: int
) -> dict:
    """
    Collect and parallel-load JSON data from all grids in grid_list.
    """
    rng = random.Random(seed)
    all_json_paths = []

    for gridname in grid_list:
        path_grid = os.path.join(local_dir, gridname)
        group_list = [g for g in os.listdir(path_grid) if g.startswith('group')]
        rng.shuffle(group_list)

        for group in group_list:
            path_group = os.path.join(path_grid, group)
            json_list = [fname for fname in os.listdir(path_group) if fname.endswith('.json')]
            rng.shuffle(json_list)

            n_sample_files = math.ceil(len(json_list) * data_frac)
            json_list = json_list[:n_sample_files]
            for fname in json_list:
                all_json_paths.append(os.path.join(path_group, fname))

    rng.shuffle(all_json_paths)

    combined_dataset = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_read_json, fpath) for fpath in all_json_paths]
        for f in as_completed(futures):
            data_part = f.result()
            combined_dataset.update(data_part)

    return combined_dataset


def _read_json(fpath: str) -> dict:
    with open(fpath, 'r') as fp:
        return json.load(fp)


def _set_nodelevel_values(
    n: int, 
    datapoint_dict: dict, 
    load_buses: dict, 
    shunt_buses: int
) -> Tuple:
    Sd_re_n = np.zeros(n, dtype=np_dtype)
    Sd_im_n = np.zeros(n, dtype=np_dtype)
    Ys_re_n = np.zeros(n, dtype=np_dtype)
    Ys_im_n = np.zeros(n, dtype=np_dtype)
    vl_n    = np.zeros(n, dtype=np_dtype)
    vu_n    = np.zeros(n, dtype=np_dtype)
    bustype_n  = np.zeros(n, dtype=np.intc)
    basekV_n   = np.zeros(n, dtype=np.intc)

    # Fill node values
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['bus']):
        basekV_n[idx]  = values[0]
        bustype_n[idx] = values[1]  # PQ(1), PV(2), REF(3), etc.
        vl_n[idx]      = values[2]  # p.u.
        vu_n[idx]      = values[3]

    # Load data
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['load']):
        Sd_re_n[load_buses[idx]] += values[0]
        Sd_im_n[load_buses[idx]] += values[1]

    # Shunt data
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['shunt']):
        Ys_im_n[shunt_buses[idx]] += values[0]
        Ys_re_n[shunt_buses[idx]] += values[1]

    return (Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, basekV_n, vl_n, vu_n, bustype_n)


def _set_generatorlevel_values(
    n: int, 
    n_g: int, 
    datapoint_dict: dict, 
    generator_buses: list
) -> Tuple:
    # Accumulate generator constraints at node level
    Sgl_re_n   = np.zeros(n, dtype=np_dtype)
    Sgl_im_n   = np.zeros(n, dtype=np_dtype)
    Sgu_re_n   = np.zeros(n, dtype=np_dtype)
    Sgu_im_n   = np.zeros(n, dtype=np_dtype)
    c0g_n      = np.zeros(n, dtype=np_dtype)
    c1g_n      = np.zeros(n, dtype=np_dtype)
    c2g_n      = np.zeros(n, dtype=np_dtype)
    num_gen_n  = np.zeros(n, dtype=np.intc)

    # Generator-level arrays
    Sl_re_g = np.zeros(n_g, dtype=np_dtype)
    Sl_im_g = np.zeros(n_g, dtype=np_dtype)
    Su_re_g = np.zeros(n_g, dtype=np_dtype)
    Su_im_g = np.zeros(n_g, dtype=np_dtype)
    c0_g    = np.zeros(n_g, dtype=np_dtype)
    c1_g    = np.zeros(n_g, dtype=np_dtype)
    c2_g    = np.zeros(n_g, dtype=np_dtype)
    mbase_g = np.zeros(n_g, dtype=np_dtype)
    pg_init_g = np.zeros(n_g, dtype=np_dtype)
    qg_init_g = np.zeros(n_g, dtype=np_dtype)

    gen_bus_g = np.array(generator_buses, dtype=np.intc)

    for idx, values in enumerate(datapoint_dict['grid']['nodes']['generator']):
        mbase_g[idx]   = values[0]
        pg_init_g[idx] = values[1]
        Sl_re_g[idx]   = values[2]
        Su_re_g[idx]   = values[3]
        qg_init_g[idx] = values[4]
        Sl_im_g[idx]   = values[5]
        Su_im_g[idx]   = values[6]
        # values[7] is vm_init_n, unused here
        c2_g[idx]      = values[8]
        c1_g[idx]      = values[9]
        c0_g[idx]      = values[10]

    for gen_id, node_id in enumerate(gen_bus_g):
        Sgl_re_n[node_id] += Sl_re_g[gen_id]
        Sgl_im_n[node_id] += Sl_im_g[gen_id]
        Sgu_re_n[node_id] += Su_re_g[gen_id]
        Sgu_im_n[node_id] += Su_im_g[gen_id]
        # Notice c2_g, c1_g, c0_g are being reversed on purpose.
        c0g_n[node_id]    += c2_g[gen_id]
        c1g_n[node_id]    += c1_g[gen_id]
        c2g_n[node_id]    += c0_g[gen_id]
        num_gen_n[node_id] += 1

    return (
        Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g,
        pg_init_g, qg_init_g, gen_bus_g,
        Sgl_re_n, Sgl_im_n, Sgu_re_n, Sgu_im_n, c0g_n, c1g_n, c2g_n, num_gen_n
    )


def _set_edgelevel_values(
    n_e: int, 
    datapoint_dict: dict, 
    Ys_re_n: np.ndarray, 
    Ys_im_n: np.ndarray
) -> Tuple:
    # Line buses
    ij_line = np.column_stack(
        (
            datapoint_dict['grid']['edges']['ac_line']['senders'],
            datapoint_dict['grid']['edges']['ac_line']['receivers']
        )
    )
    # Transformer buses
    ij_transformer = np.column_stack(
        (
            datapoint_dict['grid']['edges']['transformer']['senders'],
            datapoint_dict['grid']['edges']['transformer']['receivers']
        )
    )

    ij_e = np.vstack((ij_line, ij_transformer)).astype(np.intc)
    Y_re_e = np.zeros(n_e, dtype=np_dtype)
    Y_im_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ij_im_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ijR_im_e = np.zeros(n_e, dtype=np_dtype)
    T_mag_e = np.ones(n_e, dtype=np_dtype)
    T_ang_e = np.zeros(n_e, dtype=np_dtype)
    su_e = np.zeros(n_e, dtype=np_dtype)
    vangl_e = np.zeros(n_e, dtype=np_dtype)
    vangu_e = np.zeros(n_e, dtype=np_dtype)

    # For completeness, these arrays remain but are not explicitly filled:
    Yc_ij_re_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ijR_re_e = np.zeros(n_e, dtype=np_dtype)

    # Fill line features
    idx = -1
    for line_vals in datapoint_dict['grid']['edges']['ac_line']['features']:
        idx += 1
        vangl_e[idx]     = line_vals[0]
        vangu_e[idx]     = line_vals[1]
        Yc_ij_im_e[idx]  = line_vals[2]
        Yc_ijR_im_e[idx] = line_vals[3]
        r = line_vals[4]
        x = line_vals[5]
        denom = r**2 + x**2
        Y_re_e[idx] = r / denom
        Y_im_e[idx] = -x / denom
        su_e[idx]   = line_vals[6]

    # Fill transformer features
    for tran_vals in datapoint_dict['grid']['edges']['transformer']['features']:
        idx += 1
        vangl_e[idx]     = tran_vals[0]
        vangu_e[idx]     = tran_vals[1]
        r = tran_vals[2]
        x = tran_vals[3]
        denom = r**2 + x**2
        Y_re_e[idx] = r / denom
        Y_im_e[idx] = -x / denom
        su_e[idx]   = tran_vals[4]
        T_mag_e[idx] = tran_vals[7]
        T_ang_e[idx] = tran_vals[8]
        Yc_ij_im_e[idx]  = tran_vals[9]
        Yc_ijR_im_e[idx] = tran_vals[10]

    # Convert Y to polar
    Y_mag_e, Y_ang_e = _rectangle_to_polar(Y_re_e, Y_im_e)

    # Build up node admittance
    Y_re_n_new = Ys_re_n.copy()
    Y_im_n_new = Ys_im_n.copy()
    for branch_k in range(len(ij_e)):
        i_node = ij_e[branch_k, 0]
        j_node = ij_e[branch_k, 1]
        Y_re_n_new[i_node] += Y_re_e[branch_k]
        Y_re_n_new[j_node] += Y_re_e[branch_k]
        Y_im_n_new[i_node] += Y_im_e[branch_k]
        Y_im_n_new[j_node] += Y_im_e[branch_k]

    return (
        ij_e, Y_re_e, Y_im_e, Yc_ij_im_e, Yc_ijR_im_e,
        T_mag_e, T_ang_e, su_e, vangl_e, vangu_e,
        Yc_ij_re_e, Yc_ijR_re_e, Y_re_n_new, Y_im_n_new, Y_mag_e, Y_ang_e
    )


def _rectangle_to_polar(X_re, X_im):
    """
    Transform from rectangular to polar, avoiding zero-div by a small offset.
    """
    small_number = 1.e-10
    X_mag_np = np.sqrt(X_re**2 + X_im**2)
    # Keep user logic: avoid pure arctan2
    X_ang_np = np.arctan(X_im / (X_re + small_number))
    return X_mag_np, X_ang_np


def _to_numpy_dict(data_dict: dict):
    """
    Convert each array-like or scalar in data_dict.
    """
    for key, value in data_dict.items():
        if isinstance(value, (int, float)):
            data_dict[key] = np.array([value], dtype=np_dtype)
    return data_dict


def obj_gen_cost(Sg_re_g, c2_g, c1_g, c0_g):
    return c2_g * Sg_re_g**2 + c1_g * Sg_re_g + c0_g

def eq_pbalance_re(Sg_re_n, Sd_re_n, Ys_re_n, V_mag_n, Sij_re_n, SijR_re_n):
    return Sg_re_n - Sd_re_n - Ys_re_n * V_mag_n**2 - Sij_re_n - SijR_re_n

def eq_pbalance_im(Sg_im_n, Sd_im_n, Ys_im_n, V_mag_n, Sij_im_n, SijR_im_n):
    return Sg_im_n - Sd_im_n + Ys_im_n * V_mag_n**2 - Sij_im_n - SijR_im_n

def ineq_lower_box(x_value, x_lower):
    return x_lower - x_value

def ineq_upper_box(x_value, x_upper):
    return x_value - x_upper

def eq_difference(x_value, x_true_value):
    return x_value - x_true_value

def _create_taskdescription():
    """Contains natural language description of task. Placeholder."""

    task_description = """
    Given the specifications of an electric power system at a single point in 
    time, the goal in solving Alternating Current Optimal Power Flow (ACOPF)
    problem instances with ML is to predict the most cost-effective power output 
    for a set of generators while ensuring compliance with physical constraints.

    This problem, an NP-hard optimization challenge, is traditionally solved 
    using non-ML methods by power system operators. Enhancing its solution speed
    and optimality through ML can significantly improve the planning and 
    real-time dispatch of high-renewable power systems, which inherently face 
    greater supply uncertainty than today's fossil fuel-dominated power systems.

    The fundamental physical law that each solution has to satisfy is 
    Kirchhoff's Current Law.

    We define six sub-task datasets, each designed to evaluate the generalization 
    capabilities of solution methods across varying power system topologies. The 
    overall dataset comprises problem instances from ten different power system 
    topologies, with distinct training, validation, and testing splits based on 
    topology size. Each sub-task involves training and validation on a subset of 
    either small, medium, or large topologies while testing on a different subset, 
    enabling a structured assessment of scalability and transferability. These 
    are:

    train_small_test_medium: Training and validation data from 14-, 30-, and 
    57-bus systems, tested on 118-, 500-, and 2,000-bus systems.

    train_small_test_large: Training and validation data from 14-, 30-, and 
    57-bus systems, tested on 4,661-, 6,470-, 10,000-, and 13,659-bus systems.

    train_medium_test_small: Training and validation data from 118-, 500-, and 
    2,000-bus systems, tested on 14-, 30-, and 57-bus systems.

    train_medium_test_large: Training and validation data from 118-, 500-, and 
    2,000-bus systems, tested on 4,661-, 6,470-, 10,000-, and 13,659-bus systems.

    train_large_test_small: Training and validation data from 4,661-, 6,470-, 
    10,000-, and 13,659-bus systems, tested on 14-, 30-, and 57-bus systems.

    train_large_test_medium: Training and validation data from 4,661-, 6,470-, 
    10,000-, and 13,659-bus systems, tested on 118-, 500-, and 2,000-bus systems.
    """

    return task_description



def _create_subtaskdescription(subtask_name: str):
    """Contains natural language description of subtask. Placeholder."""

    subtask_description = f"""
    Here, we are solving instances of the {subtask_name} subtask.
    """.format(subtask_name)
    
    return subtask_description


