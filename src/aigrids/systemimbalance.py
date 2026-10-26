"""Loads standardized subtasks for SystemImbalance.

See for a full documentation of task and subtask data:
https://github.com/aigrids/aigrids/blob/master/docs/systemimbalance.md

"""
import os
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np

AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_volume_predict1h',
    'odd_space_volume_predict1h',
    'odd_spacetime_volume_predict1h',
    'odd_time_price_predict1h'
]

# Local time zone of the German electricity market.
TIMEZONE = 'Europe/Berlin'

# Historic data time steps.
HISTORIC_WINDOW = 96 # 4 * 24, one day data in 15 minute resolution.

# Predicted time steps.
PRED_WINDOW = 4 # one hour in 15 minute resolution.

# Delay in time steps after which a quarter hour of a historic series is
# published. The transmission system operators publish the operational system
# imbalance 15 minutes after the end of each quarter hour, and the estimate of
# the balancing energy price 30 minutes after it. The historic window of each
# series therefore ends this many time steps before the predicted hour.
PUBLICATION_DELAY = {
    'imbalance': 1,
    'imbalance_other': 1,
    'price_imb': 2
}

# Step size between consecutive sliding windows, in time steps. Striding by
# the prediction window avoids overlapping labels between data points.
WINDOW_STRIDE = PRED_WINDOW

# The standardized split ratio for the entire dataset: train, val, test
SPLIT_RATIO = (0.5, 0.1, 0.4)

# Control zone that is only used for testing in odd_space and odd_spacetime.
TEST_ZONE = 'TransnetBW'

# Day-ahead information that is known for the predicted hour.
FEATURE_COLS = [
    'load_fc',
    'gen_fc',
    'wind_on_fc',
    'wind_off_fc',
    'pv_fc',
    'price_da'
]

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
        Number of parallel workers. Not used for this task.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    ----------
    Dict[str, List[Dict[str, Any]]]
        A dictionary with keys ['train_data', 'val_data', 'test_data'].

    """
    if subtask_name not in AVAIL_SUBTASKNAMES_LIST:
        raise ValueError(f"Unknown subtask name: {subtask_name}")

    # load dataset
    df_dataset, label_col, hist_cols, feature_cols = _load_df(
        local_dir, subtask_name
    )

    # create dataset splits
    train_data, val_data, test_data = _split_data(
        df_dataset,
        subtask_name,
        label_col,
        hist_cols,
        feature_cols,
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

def _load_df(local_dir: str, subtask_name: str):
    """
    Load DataFrame containing csv data of requested subtask.

    Parameters
    ----------
    local_dir : str
        Local path containing the subtask data.
    subtask_name : str
        One of the recognized subtask names (see AVAIL_SUBTASKNAMES_LIST).

    Returns
    ----------
    df_dataset : pd.DataFrame
        Pandas DataFrame with one row per zone and quarter hour.
    label_col : str
        Name of column containing labels.
    hist_cols : List[str]
        Names of columns provided for the historic window.
    feature_cols : List[str]
        Names of columns containing day-ahead features.

    """
    path_zones = os.path.join(local_dir, 'systemimbalance_zones.csv')
    df_dataset = pd.read_csv(path_zones)

    if 'volume' in subtask_name:
        # The system imbalance of the other three control zones is published
        # at the same time as that of the zone itself.
        n_zones = df_dataset['zone'].nunique()
        imbalance_total = df_dataset.groupby('Tmstamp')['imbalance'].transform(
            lambda x: x.sum(min_count=n_zones)
        )
        df_dataset['imbalance_other'] = imbalance_total - df_dataset['imbalance']
        label_col = 'imbalance'
        hist_cols = ['imbalance', 'imbalance_other']
        feature_cols = FEATURE_COLS

    else:
        # The balancing energy price is the same for all of Germany. The
        # system imbalance of Germany is the sum of all control zones.
        df_imbalance = df_dataset.groupby('Tmstamp')['imbalance'].sum(
            min_count=df_dataset['zone'].nunique()
        ).reset_index()

        path_national = os.path.join(local_dir, 'systemimbalance_national.csv')
        df_dataset = df_imbalance.merge(pd.read_csv(path_national), on='Tmstamp')
        df_dataset['zone'] = 'DE'
        label_col = 'price_imb'
        hist_cols = ['imbalance', 'price_imb']
        feature_cols = FEATURE_COLS + ['net_export_sched']

    df_dataset['Tmstamp'] = pd.to_datetime(
        df_dataset['Tmstamp'], utc=True
    ).dt.tz_convert(TIMEZONE)

    return df_dataset, label_col, hist_cols, feature_cols

def _split_data(
    df_dataset: pd.DataFrame,
    subtask_name: str,
    label_col: str,
    hist_cols: List[str],
    feature_cols: List[str],
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
        Dataframe containing system imbalance, day-ahead features and labels
        of all zones.

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

    # Every data point is one zone and one predicted hour. Missing values are
    # kept.
    records_dict = _make_records(df_dataset, label_col, hist_cols, feature_cols)

    stamps = sorted({stamp for (_, stamp) in records_dict})
    n_t = len(stamps)
    TRN, VAL, _ = SPLIT_RATIO
    t1, t2 = int(n_t * TRN), int(n_t * (TRN + VAL))
    trn_t, val_t, tst_t = set(stamps[:t1]), set(stamps[t1:t2]), set(stamps[t2:])

    # odd_space only: with three training zones, validation data is taken
    # from the last time stamps of these.
    trn_t_space = set(stamps[:int(n_t * TRN / (TRN + VAL))])

    train_data, val_data, test_data = [], [], []
    for (zone, stamp), record in records_dict.items():

        if subtask_name.startswith("odd_time"):
            # split on *timestamps* only
            if stamp in trn_t:
                train_data.append(record)
            elif stamp in val_t:
                val_data.append(record)
            else:
                test_data.append(record)

        elif subtask_name.startswith("odd_spacetime"):
            # split on zones *and* timestamps
            if zone != TEST_ZONE and stamp in trn_t:
                train_data.append(record)
            elif zone != TEST_ZONE and stamp in val_t:
                val_data.append(record)
            elif zone == TEST_ZONE and stamp in tst_t:
                test_data.append(record)

        elif subtask_name.startswith("odd_space"):
            # split on *zones* only
            if zone == TEST_ZONE:
                test_data.append(record)
            elif stamp in trn_t_space:
                train_data.append(record)
            else:
                val_data.append(record)

    # Thin datasets by subsampling *generated windows*, not raw rows, so
    # each remaining window still spans a valid contiguous time range.
    if 0 < data_frac < 1:
        train_data = _subsample_records(train_data, data_frac, rng)
        val_data   = _subsample_records(val_data, data_frac, rng)
        test_data  = _subsample_records(test_data, data_frac, rng)

    # Thin the *train* windows
    if 0 < train_frac < 1:
        train_data = _subsample_records(train_data, train_frac, rng)

    return train_data, val_data, test_data

def _make_records(
    df_dataset: pd.DataFrame,
    label_col: str,
    hist_cols: List[str],
    feature_cols: List[str]
) -> Dict[tuple, Dict[str, Any]]:
    """
    Create one record per zone and sliding window. A window consists of
    the historic columns of the historic window, each ending at its
    publication delay before the prediction window, and of the day-ahead
    features and labels of the following prediction window.
    """
    records_dict: Dict[tuple, Dict[str, Any]] = {}

    for zone, g in df_dataset.groupby('zone', sort=True):
        g = g.sort_values('Tmstamp').reset_index(drop=True)

        # fast access arrays
        hist_data = g[hist_cols].to_numpy(dtype=np.float32)
        feat_data = g[feature_cols].to_numpy(dtype=np.float32)
        labels    = g[label_col].to_numpy(dtype=np.float32)
        stamps    = [stamp.isoformat() for stamp in g['Tmstamp']]

        delays = [PUBLICATION_DELAY[col] for col in hist_cols]
        max_start = len(g) - HISTORIC_WINDOW - PRED_WINDOW
        for start in range(0, max_start + 1, WINDOW_STRIDE):
            # historic windows must not start before the first time step
            if start < max(delays):
                continue
            h_end = start + HISTORIC_WINDOW
            p_end = h_end + PRED_WINDOW

            # time stamp of the first predicted quarter hour
            stamp = stamps[h_end]

            # build feature dict
            feat: Dict[str, Any] = {
                "Tmstamp": stamp,
                "zone":    zone
            }
            # historic values that are published before the predicted hour
            for col_idx, col in enumerate(hist_cols):
                d = delays[col_idx]
                feat[col] = hist_data[start - d:h_end - d, col_idx].tolist()

            pred_block = feat_data[h_end:p_end]   # [PRED_WINDOW, d]
            for col_idx, col in enumerate(feature_cols):
                feat[col] = pred_block[:, col_idx].tolist()

            lbl: Dict[str, Any] = {
                label_col: labels[h_end:p_end].tolist()
            }
            records_dict[(zone, stamp)] = {"features": feat, "label": lbl}

    return records_dict

def _subsample_records(
    records: List[Dict[str, Any]],
    frac: float,
    rng: np.random.Generator
) -> List[Dict[str, Any]]:
    """Randomly keep a fraction of already-built window records."""

    if not records:
        return records

    k = max(1, int(len(records) * frac))
    keep_idx = rng.choice(len(records), size=k, replace=False)

    return [records[i] for i in keep_idx]

def _create_taskdescription():
    """Contains natural language description of task. Placeholder."""

    task_description = """
    Given the system imbalance of a control zone and the summed system
    imbalance of the other control zones for 24 hours, as far as they are
    published before the predicted hour, together with the day-ahead
    forecasts of electric load, total generation, wind and solar generation,
    and the day-ahead wholesale electricity prices for the next hour, the goal
    of the SystemImbalance challenge is to predict the quarter-hourly system
    imbalance of the control zone for the next hour. A second set of data
    points is formed to predict the quarter-hourly balancing energy price in
    Germany instead.

    Solving this task and accurately forecasting system imbalances enables
    transmission system operators to activate balancing reserves more
    efficiently, and allows market parties to reduce their own imbalances.
    This lowers the cost of integrating higher shares of fluctuating wind and
    solar generation into existing grids.

    All data is published by Bundesnetzagentur | SMARD.de.

    We distinguish four sub-tasks, a first set containing data points formed to
    predict the system imbalance of the four German control zones and a second
    set containing data points formed to predict the balancing energy price:

    odd_time_volume_predict1h: Predict the system imbalance of a control zone
    for the next hour, with test time stamps not present in training data.

    odd_space_volume_predict1h: Predict the system imbalance of a control zone
    for the next hour, with the test control zone not present in training
    data.

    odd_spacetime_volume_predict1h: Predict the system imbalance of a control
    zone for the next hour, with test time stamps and the test control zone
    not present in training data.

    odd_time_price_predict1h: Predict the balancing energy price in Germany
    for the next hour from the system imbalance and the balancing energy
    price of Germany for 24 hours, as far as they are published before the
    predicted hour, with test time stamps not present in training data.

    """

    return task_description

def _create_subtaskdescription(subtask_name: str):
    """Contains natural language description of subtask. Placeholder."""

    subtask_description = f"""
    Here, we are solving instances of the {subtask_name} subtask.
    """.format(subtask_name)

    return subtask_description
