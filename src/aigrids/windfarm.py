"""Loads standardized subtasks for WindFarm.

See for a full documentation of task and subtask data:
https://github.com/AI-grids/aidotgrids/blob/master/docs/windfarm.md

"""
import os
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import gc

AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_predict48h',
    'odd_space_predict48h',
    'odd_spacetime_predict48h',
    'odd_time_predict72h',
    'odd_space_predict72h',
    'odd_spacetime_predict72h'
]

# Historic data time steps.
HISTORIC_WINDOW = 144 # 6 * 24, one day data in 10 minute resolution.

# The standardized split ratio for the entire dataset: train, val, test
SPLIT_RATIO = (0.5, 0.1, 0.4)

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

    Parameters
    ----------
    local_dir : str
        Local path containing the subtask data.
    subtask_name : str
        One of the recognized subtask names (see AVAIL_SUBTASKNAMES_LIST).
    data_frac : Union[int, float]
        Overall fraction of samples to keep from full dataset.
    train_frac : Union[int, float]
        Fraction of the standardized training split to actually use.
    max_workers : int
        Number of parallel workers for loading data from HDF5.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    ----------
    Dict[str, List[Dict[str, Any]]]
        A dictionary with keys ['train_data', 'val_data', 'test_data'].

    """
    if subtask_name not in AVAIL_SUBTASKNAMES_LIST:
        raise ValueError(f"Unknown subtask name: {subtask_name}")

    # load and merge dataset
    df_dataset = _load_and_merge_df(local_dir)

    # create dataset splits
    train_data, val_data, test_data = _split_data(
        df_dataset, 
        subtask_name,
        data_frac, 
        train_frac,
        seed
    )

    # Load natural language descriptions
    task_description    = _create_taskdescription()
    subtask_description = _create_subtaskdescription(subtask_name)

    subtask_data = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'task_description': task_description,
        'subtask_description': subtask_description
    }

    return subtask_data

def _split_data(
    df_dataset: pd.DataFrame,
    subtask_name: str,
    data_frac: Union[int, float],
    train_frac: Union[int, float],
    seed: int
):
    """
    Split `df_dataset` into train/val/test lists of dicts according to the 
    odd-* subtasks.

    Parameters
    ----------
    df_dataset : pd.DataFrame
        Merged dataframe containing all scada and location data of turbines.

    Returns
    ----------
    train_data : List
        Training data.
    val_data : List
        Validation data.
    test_data : List
        Testing data.

    """
    rng = np.random.default_rng(seed)

    # Determine prediction window length
    if subtask_name.endswith("48h"):
        pred_window = 288        # 48 h  at 10-min resolution
    elif subtask_name.endswith("72h"):
        pred_window = 432        # 72 h

    # Thin the raw dataframe
    if 0 < data_frac < 1:
        df_dataset = df_dataset.sample(frac=data_frac, random_state=seed)

    # keep dtypes tidy
    df_dataset["Tmstamp"] = pd.to_datetime(df_dataset["Tmstamp"])
    df_dataset.sort_values(["TurbID", "Tmstamp"], inplace=True)

    # columns
    STATIC_COLS = ["Tmstamp", "x", "y", "Ele"]
    SEQ_COLS    = [
        c for c in df_dataset.columns
        if c not in ("TurbID", "Patv", *STATIC_COLS)
    ]


    TRN, VAL, TST = SPLIT_RATIO
    rng_split = np.random.default_rng(seed)
    
    if subtask_name.startswith("odd_time"):
        # split on *timestamps* only
        stamps = df_dataset["Tmstamp"].drop_duplicates().sort_values()
        n_t    = len(stamps)
        t1, t2 = int(n_t * TRN), int(n_t * (TRN + VAL))
        train_mask = df_dataset["Tmstamp"].isin(stamps[:t1])
        val_mask   = df_dataset["Tmstamp"].isin(stamps[t1:t2])
        test_mask  = df_dataset["Tmstamp"].isin(stamps[t2:])

    elif subtask_name.startswith("odd_space"):
        # split on *turbines* only
        turbs = np.sort(df_dataset["TurbID"].unique())
        rng_split.shuffle(turbs)
        n_tb   = len(turbs)
        t1, t2 = int(n_tb * TRN), int(n_tb * (TRN + VAL))
        trn_tb, val_tb, tst_tb = turbs[:t1], turbs[t1:t2], turbs[t2:]
        train_mask = df_dataset["TurbID"].isin(trn_tb)
        val_mask   = df_dataset["TurbID"].isin(val_tb)
        test_mask  = df_dataset["TurbID"].isin(tst_tb)

    elif subtask_name.startswith("odd_spacetime"):
        # split on turbines *and* timestamps
        turbs = np.sort(df_dataset["TurbID"].unique())
        rng_split.shuffle(turbs)
        n_tb   = len(turbs)
        t1_tb, t2_tb = int(n_tb * TRN), int(n_tb * (TRN + VAL))
        trn_tb, val_tb, tst_tb = turbs[:t1_tb], turbs[t1_tb:t2_tb], turbs[t2_tb:]

        stamps = df_dataset["Tmstamp"].drop_duplicates().sort_values()
        n_t    = len(stamps)
        t1_t, t2_t = int(n_t * TRN), int(n_t * (TRN + VAL))
        trn_t, val_t, tst_t = stamps[:t1_t], stamps[t1_t:t2_t], stamps[t2_t:]

        train_mask = df_dataset["TurbID"].isin(trn_tb) & df_dataset["Tmstamp"].isin(trn_t)
        val_mask   = df_dataset["TurbID"].isin(val_tb) & df_dataset["Tmstamp"].isin(val_t)
        test_mask  = df_dataset["TurbID"].isin(tst_tb) & df_dataset["Tmstamp"].isin(tst_t)

    splits = {
        "train": df_dataset.loc[train_mask],
        "val":   df_dataset.loc[val_mask],
        "test":  df_dataset.loc[test_mask],
    }


    def _make_records(df_split: pd.DataFrame) -> List[Dict[str, Any]]:
        """Helper function. """
        
        records: List[Dict[str, Any]] = []
        for tid, g in df_split.groupby("TurbID"):
            g = g.sort_values("Tmstamp").reset_index(drop=True)

            # fast access arrays
            seq_data = g[SEQ_COLS].to_numpy(dtype=np.float32)
            labels   = g["Patv"].to_numpy(dtype=np.float32)
            stamps   = g["Tmstamp"].to_numpy()

            n_rows = len(g)
            max_start = n_rows - HISTORIC_WINDOW - pred_window
            if max_start < 0:
                continue  # not enough history for this turbine

            for start in range(max_start + 1):
                h_end = start + HISTORIC_WINDOW
                p_end = h_end + pred_window

                # build feature dict
                feat: Dict[str, Any] = {
                    "Tmstamp": pd.Timestamp(stamps[h_end - 1]).isoformat(),
                    "x":       float(g.at[0, "x"]),
                    "y":       float(g.at[0, "y"]),
                    "Ele":     float(g.at[0, "Ele"]),
                }
                hist_block = seq_data[start:h_end]   # [HISTORIC_WINDOW, d]
                for col_idx, col in enumerate(SEQ_COLS):
                    feat[col] = hist_block[:, col_idx].tolist()

                label_seq = labels[h_end:p_end].tolist()

                lbl: Dict[str, Any] = {
                    "Patv": label_seq
                }
                records.append({"features": feat, "label": lbl})

        return records

    # Generate records for every split
    train_data = _make_records(splits["train"])
    val_data   = _make_records(splits["val"])
    test_data  = _make_records(splits["test"])

    # Thin the *train* windows
    if 0 < train_frac < 1:
        k = int(len(train_data) * train_frac)
        train_data = rng.choice(train_data, size=k, replace=False).tolist()

    return train_data, val_data, test_data


def _load_and_merge_df(local_dir: str) -> pd.DataFrame:
    """
    Load and merge DataFrame containing csv data.

    Parameters
    ----------
    local_dir : str
        Local path containing the subtask data.

    Returns
    ----------
    df_dataset : pd.DataFrame
        Merged Pandas DataFrame containing all data.

    """
    # load SCADA data
    path_scada = os.path.join(local_dir, 'sdwpf_2001_2112_full.csv')
    df_scada = pd.read_csv(path_scada)

    # load location data
    path_loc = os.path.join(local_dir, 'sdwpf_turb_location_elevation.csv')
    df_loc = pd.read_csv(path_loc)

    # merge dataframes by turbine IDs.
    df_dataset = df_scada.merge(df_loc)

    # clean up
    del df_scada, df_loc
    gc.collect()

    return df_dataset

def _create_taskdescription():
    """Contains natural language description of task. Placeholder."""

    task_description = """
    Given two days of historic power generation data for each turbine in a wind
    farm, along with the relative geographic positions of the turbines, the goal
    of the WindFarm challenge is to predict the active power output profile of 
    the entire farm for the next 48 hours.

    Solving this task and accurately forecasting wind farm generation enables 
    power system operators to better manage the inherent uncertainty of wind 
    energy. This, in turn, supports the optimal use of storage capacities and 
    flexible loads, enhances the planning and dispatch of electric transmission 
    systems, and facilitates the faster integration of higher shares of wind 
    energy into existing grids.

    Fluid mechanics constitutes the main physical foundation underlying 
    solutions to this task.

    We distinguish six sub-tasks, a first set containing data points formed to 
    predict wind power output for 48 hours into the future and a second set 
    containing data points formed to predict wind power output for 72 hours:

    odd_time_predict48h: Predict wind power profiles for 48 hours into the 
    future, with test time stamps not present in training data.

    odd_space_predict48h: Predict wind power profiles for 48 hours into the 
    future, with test turbine locations not present in training data.

    odd_spacetime_predict48h: Predict wind power profiles for 48 hours into the 
    future, with test time stamps and turbines not present in training data.

    odd_time_predict72h: Predict wind power profiles for 72 hours into the 
    future, with test time stamps not present in training data.

    odd_space_predict72h: Predict wind power profiles for 72 hours into the 
    future, with test turbines not present in training data.

    odd_spacetime_predict72h: Predict wind power profiles for 72 hours into the 
    future, with test time stamps and turbines not present in training data.

    """

    return task_description

def _create_subtaskdescription(subtask_name: str):
    """Contains natural language description of subtask. Placeholder."""

    subtask_description = f"""
    Here, we are solving instances of the {subtask_name} subtask.
    """.format(subtask_name)
    
    return subtask_description

    