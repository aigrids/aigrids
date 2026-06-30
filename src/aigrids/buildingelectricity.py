"""Loads requested subtask for BuildingElectricity.

"""
import os
from PIL import Image
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
import gc



AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_buildings92',
    'odd_space_buildings92',
    'odd_spacetime_buildings92',
    'odd_time_wload_buildings92',
    'odd_space_wload_buildings92',
    'odd_spacetime_wload_buildings92',
    'odd_time_buildings451',
    'odd_space_buildings451',
    'odd_spacetime_buildings451',
    'odd_time_wload_buildings451',
    'odd_space_wload_buildings451',
    'odd_spacetime_wload_buildings451'
]

ZOOM_LEVEL_LIST = [
    'zoom1',
    'zoom2',
    'zoom3'
]

IMAGE_TYPE_LIST = [
    'aspect',
    'ortho',
    'relief',
    'slope'
]

HISTORIC_WINDOW_SIZE = 96
LABEL_WINDOW_SIZE = 96

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
    -------
    Dict[str, List[Dict[str, Any]]]
        A dictionary with keys ['train_data', 'val_data', 'test_data'].

    """
    if subtask_name not in AVAIL_SUBTASKNAMES_LIST:
        raise ValueError(f"Unknown subtask name: {subtask_name}")

    # exdend local_dir with corresponding profiles
    if subtask_name.endswith('92'):
        local_dir = os.path.join(local_dir, 'profiles_92')
    elif subtask_name.endswith('451'):
        local_dir = os.path.join(local_dir, 'profiles_451')
    else:
        raise ValueError('Check subtask handling. Naming not consistent!')

    # load electric load profiles
    (
        df_loads, 
        building_to_cluster, 
        time_stamps
    ) = _load_electric_load_profiles(local_dir)

    # load building images
    building_images = _load_building_images(local_dir)

    # load cluster images
    cluster_images = _load_cluster_images(local_dir)

    # load meteo data
    meteo_dict = _load_meteo_data(local_dir)

    # pair data
    paired_dataset = _pair_data(
        df_loads,
        meteo_dict,
        building_to_cluster, 
        time_stamps,
        subtask_name
    )

    del df_loads, building_to_cluster, time_stamps, meteo_dict
    gc.collect()

    # split data into train, val, test data
    train_data, val_data, test_data = _split_data(
        paired_dataset, 
        subtask_name,
        seed
    )

    # Load natural language descriptions
    task_description    = _create_taskdescription()
    subtask_description = _create_subtaskdescription(subtask_name)
    data_description    = _create_datadescription()

    # bundle to training validation and testing data
    subtask_data = {
        'task_description': task_description,
        'subtask_description': subtask_description,
        'data_description': data_description,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'building_images': building_images,
        'cluster_images': cluster_images
    }

    return subtask_data


def _pair_data(
    df_loads: pd.DataFrame,
    meteo_dict: dict,
    building_to_cluster: dict, 
    time_stamps: pd.DataFrame,
    subtask_name: str
) -> dict:
    """
    Pair all data components into single data points and return as dictionary.

    Parameters
    ----------
    df_loads : pd.DataFrame
        Dataframe containing all load profiles in each column, with columns
        being building IDs.
    meteo_dict : dict
        Dictionary of meteorological Dataframe values. One entry per Cluster.
    building_to_cluster : dict
        Mapping from building IDs to cluster IDs.
        
    Returns
    -------
    paired_dataset : dict

    """
    # fill this
    paired_dataset = {}
    datapoint_counter = 0

    # iterate over all columns/building IDs
    for building_id in df_loads.columns:
        # set cluster ID
        cluster_id = building_to_cluster[int(building_id)]

        # set load profile
        load_profile = df_loads[building_id]

        # iterate over loads in window sizes.
        for i in range(
            HISTORIC_WINDOW_SIZE, 
            len(load_profile), 
            HISTORIC_WINDOW_SIZE + LABEL_WINDOW_SIZE
        ):
            # set historic load
            historic_load = load_profile.iloc[i-HISTORIC_WINDOW_SIZE:i].values

            # set future load as label
            future_load = load_profile.iloc[i:i+LABEL_WINDOW_SIZE].values

            # set time stamp
            timestamp = time_stamps.iloc[i]

            # set meteo data
            meteo_df = meteo_dict[f'cluster_{cluster_id}']
            air_density = meteo_df['air_density'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            cloud_cover = meteo_df['cloud_cover'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            precipitation = meteo_df['precipitation'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            radiation_surface = meteo_df['radiation_surface'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            radiation_toa = meteo_df['radiation_toa'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            snow_mass = meteo_df['snow_mass'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            snowfall = meteo_df['snowfall'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            temperature_celsius = meteo_df['temperature'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]
            wind_speed = meteo_df['wind_speed'].iloc[
                i-HISTORIC_WINDOW_SIZE:i]

            # average values back to hourly resolution
            air_density = air_density.groupby(
                air_density.index // 4).mean().values
            cloud_cover = cloud_cover.groupby(
                cloud_cover.index // 4).mean().values
            precipitation = precipitation.groupby(
                precipitation.index // 4).mean().values
            radiation_surface = radiation_surface.groupby(
                radiation_surface.index // 4).mean().values
            radiation_toa = radiation_toa.groupby(
                radiation_toa.index // 4).mean().values
            snow_mass = snow_mass.groupby(
                snow_mass.index // 4).mean().values
            snowfall = snowfall.groupby(
                snowfall.index // 4).mean().values
            temperature_celsius = temperature_celsius.groupby(
                temperature_celsius.index // 4).mean().values
            wind_speed = wind_speed.groupby(
                wind_speed.index // 4).mean().values

            # set features
            features = {
                'historic_load': historic_load,
                'air_density': air_density,
                'cloud_cover': cloud_cover,
                'precipitation': precipitation,
                'radiation_surface': radiation_surface,
                'radiation_toa': radiation_toa,
                'snow_mass': snow_mass,
                'snowfall': snowfall,
                'temperature_celsius': temperature_celsius,
                'wind_speed': wind_speed,
                'timestamp': timestamp,
                'building_id': int(building_id),
                'cluster_id': cluster_id,
            }

            # set labels
            labels = {
                'future_load': future_load
            }

            # check if subtask is supposed to contain historic load in features
            if 'wload' not in subtask_name:
                del features['historic_load']

            # set data point
            data_point = {
                'features': features,
                'labels': labels,
            }

            # update paired data points
            paired_dataset.update(
                {datapoint_counter: data_point}
            )
            datapoint_counter += 1    

    return paired_dataset


def _split_data(paired_dataset, subtask_name, seed):
    """
    Split train/val/test data from paired_dataset based on subtask_name.
    """

    # Flatten paired data for DataFrame operations
    flat_data = []
    for item in paired_dataset.values():
        entry = {
            'features': item['features'],
            'labels': item['labels'],
            'timestamp': pd.to_datetime(item['features']['timestamp']),
            'building_id': item['features']['building_id'],
        }
        flat_data.append(entry)

    df = pd.DataFrame(flat_data)

    # Define split types
    do_time_split = ('_time_' in subtask_name)
    do_space_split = ('_space_' in subtask_name)
    do_spacetime = ('_spacetime_' in subtask_name)

    train_ratio, val_ratio, test_ratio = SPLIT_RATIO

    def to_records(df_split):
        return [{'features': row['features'], 'labels': row['labels']} for _, row in df_split.iterrows()]

    def time_based_split(df_in):
        df_sorted = df_in.sort_values(by='timestamp').reset_index(drop=True)
        n = len(df_sorted)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        df_train = df_sorted.iloc[:n_train]
        df_val = df_sorted.iloc[n_train:n_train + n_val]
        df_test = df_sorted.iloc[n_train + n_val:]
        return df_train, df_val, df_test

    def building_based_split(df_in):
        building_ids = df_in['building_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(building_ids)

        n = len(building_ids)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        bldg_train = building_ids[:n_train]
        bldg_val = building_ids[n_train:n_train + n_val]
        bldg_test = building_ids[n_train + n_val:]

        df_train = df_in[df_in['building_id'].isin(bldg_train)]
        df_val = df_in[df_in['building_id'].isin(bldg_val)]
        df_test = df_in[df_in['building_id'].isin(bldg_test)]

        return df_train, df_val, df_test

    if do_time_split and not do_space_split:
        df_train, df_val, df_test = time_based_split(df)

    elif do_space_split and not do_time_split:
        df_train, df_val, df_test = building_based_split(df)

    elif do_spacetime:
        building_ids = df['building_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(building_ids)

        n = len(building_ids)
        n_test_bldg = int(test_ratio * n)
        test_bldg = building_ids[:n_test_bldg]
        trainval_bldg = building_ids[n_test_bldg:]

        df_trainval = df[df['building_id'].isin(trainval_bldg)].sort_values(by='timestamp')
        n_tv = len(df_trainval)
        trainval_ratio = train_ratio + val_ratio
        n_train = int(train_ratio / trainval_ratio * n_tv)

        df_train = df_trainval.iloc[:n_train]
        df_val = df_trainval.iloc[n_train:]

        df_test_bldg = df[df['building_id'].isin(test_bldg)].sort_values(by='timestamp')
        n_test = int(test_ratio * len(df_test_bldg))
        df_test = df_test_bldg.iloc[-n_test:]

    else:
        raise ValueError(f"Unexpected subtask pattern: {subtask_name}")

    return to_records(df_train), to_records(df_val), to_records(df_test)


def _load_meteo_data(local_dir):
    """
    Load meteorological time series data.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    meteo_dict : dict of pd.DataFrame

    """
    # fill this
    meteo_dict = {}

    # path 
    path_meteo = os.path.join(local_dir, 'meteo_data')

    # read directory
    meteo_filename_list = os.listdir(path_meteo)

    # iterate over all filenames
    for filename in meteo_filename_list:
        # set path to file
        path_load = os.path.join(path_meteo, filename)

        # load
        meteo_file = pd.read_csv(path_load)

        # get file name key for dict
        filekey = filename.replace('.csv', '')

        # save file
        meteo_dict[filekey] = meteo_file
    
    return meteo_dict


def _load_electric_load_profiles(local_dir):
    """
    Load electric load profiles as DataFrame.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    df_loads : pd.DataFrame
    building_to_cluster : dict of int
    time_stamps : pd.Series

    """
    # set path
    path_load = os.path.join(local_dir, 'load_profiles.csv')

    # load csv
    df_loads = pd.read_csv(path_load)

    # First row is the data, first row index is probably 0
    time_stamps = df_loads.iloc[1:, 0]
    cluster_ids = df_loads.iloc[0, 1:] #skip first column (label "cluster ID")
    building_ids = df_loads.columns[1:] #skip first column (label "building ID")

    # drop cluster ID row
    df_loads.drop(labels=0, axis='index', inplace=True)

    # drop building ID column
    df_loads.drop(columns='building ID', inplace=True)

    # Create the dictionary
    building_to_cluster = dict(
        zip(building_ids.astype(int), 
        cluster_ids.astype(int))
    )

    return df_loads, building_to_cluster, time_stamps


def _load_building_images(local_dir):
    """
    Load aerial images of buildings. Use padded images.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    building_image_dict : dict of images

    """
    # fill this dictionary
    building_image_dict = {}

    # set paths and load. Use padded images.
    path_images = os.path.join(local_dir, 'building_images', 'padded')

    # list all files
    image_file_list = os.listdir(path_images)

    # iterate over all filenames.
    for filename in image_file_list:
        # check if png
        if not filename.endswith('.png'):
            continue

        # set path
        path_load = os.path.join(path_images, filename)

        # load file
        image = Image.open(path_load).convert('RGB')

        # get building ID
        building_id = filename.split('_')[1].replace('.png', '')

        # save image
        building_image_dict[int(building_id)] = image

    return building_image_dict


def _load_cluster_images(local_dir):
    """
    Load aerial images of clusters.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    zoom_image_dict : dict 
        Dictionary containig images

    """
    # fill this dictionary
    cluster_image_dict = {}

    # set path to cluster images
    path_cluster_images = os.path.join(local_dir, 'cluster_images')

    # iterate over zoom levels
    for zoom_level in ZOOM_LEVEL_LIST:
        # iterate over image types
        for image_type in IMAGE_TYPE_LIST:
            # set path to current directory
            path_images_dir = os.path.join(path_cluster_images, zoom_level, image_type)

            # list content of current directory
            image_file_list = os.listdir(path_images_dir)

            # iterate over all files in directory
            for filename in image_file_list:
                # skip if not an image
                if not filename.endswith('.png'):
                    continue

                # set cluster id
                cluster_id = int(
                    filename.replace('.png', '').replace('cluster_', '')
                )
                # set loading path
                path_load = os.path.join(path_images_dir, filename)
                # load image
                image = Image.open(path_load).convert('RGB')

                # Create nested structure if necessary
                if cluster_id not in cluster_image_dict:
                    cluster_image_dict[cluster_id] = {}
                if zoom_level not in cluster_image_dict[cluster_id]:
                    cluster_image_dict[cluster_id][zoom_level] = {}

                # set image
                cluster_image_dict[cluster_id][zoom_level][image_type] = image

    return cluster_image_dict


def _create_taskdescription():
    """Contains natural language description of task. Placeholder."""

    task_description = """
    Given the aerial image of a building and the meteorological conditions in 
    the region of that building for a past time window of 24 hours, the goal in 
    BuildingElectricity is to predict the electric load profile of single 
    buildings for a future time window of 24 hours.

    This is a short-term spatial demand forecasting challenge, where accurate 
    can significantly support the planning and dispatch of distributed renewable 
    energy sources, such as rooftop photovoltaics and micro-wind turbines, the 
    allocation and sizing of storage capacities, and the coordination of flexible 
    loads through demand response programs.

    The primary underlying pattern governing this task is the circadian rhythm 
    of human behavior, a weak but consistent physical law that shapes daily 
    electricity demand.

    We distinguish six sub-tasks, a first set containing data points from 92 
    different buildings and a second set containing data points from 451 
    different buildings:

    odd_time_buildings92: Predict electric load profiles from 92 buildings, 
    with test time stamps not present in training data.

    odd_space_buildings92: Predict electric load profiles from 92 buildings,
    with test buildings not present in training data.

    odd_spacetime_buildings92: Predict electric load profiles from 92 buildings, 
    with test time stamps and buildings not present in training data.

    odd_time_buildings451: Predict electric load profiles from 451 buildings,
    with test time stamps not present in training data.

    odd_space_buildings451: Predict electric load profiles from 451 buildings, 
    with test buildings not present in training data.

    odd_spacetime_buildings451: Predict electric load profiles from 451
    buildings, with test buildings and time stamps not present in training data.
    """

    return task_description


def _create_subtaskdescription(subtask_name: str):
    """Contains natural language description of subtask. Placeholder"""

    subtask_description = f"""
    Here, we are solving instances of the {subtask_name} subtask.
    """.format(subtask_name)
    
    return subtask_description


def _create_datadescription():
    """Contains natural language description of each data component."""

    timestamp = """
    This is a timestamp consisting of the following entries in the same order:
    year, month, day, hour, minute, second.
    """

    building_images = """
    This is an aerial image of the building with 25 cm per pixel resolution.
    """

    cluster_images_aspect = """
    This is an aerial image capturing the aspect in the greater geographic area 
    of the building in 25 m per pixel resolution. Buildings are clustered based 
    on geographic proximity and are associated with in their cluster's aerial 
    images.
    """

    cluster_images_ortho = """
    This is an aerial image of the greater geographic area of the building in 25 
    m per pixel resolution. Buildings are clustered based on geographic proximity 
    and are associated with in their cluster's aerial images.
    """

    cluster_images_relief = """
    This is an aerial image capturing the relief in the greater geographic area 
    of the building in 25 m per pixel resolution. Buildings are clustered based 
    on geographic proximity and are associated with in their cluster's aerial 
    images.
    """

    cluster_images_slope = """
    This is an aerial image capturing the slope in the greater geographic area 
    of the building in 25 m per pixel resolution. Buildings are clustered based 
    on geographic proximity and are associated with in their cluster's aerial 
    images.
    """

    air_density = """
    This is a time series measurement of air density in kg/m^3 for the past 24 
    hours in hourly resolution. 
    """

    cloud_cover = """
    This is a time series measurement of cloud cover in ratios of 0-1 for the 
    past 24 hours in hourly resolution. 
    """

    precipitation = """
    This is a time series measurement of air precipitation in mm/h for the past 
    24 hours in hourly resolution. 
    """

    radiation_surface = """
    This is a time series measurement of surface solar radiation in W/m^2 for the 
    past 24 hours in hourly resolution. 
    """

    radiation_toa = """
    This is a time series measurement of top of atmosphere solar radiation in 
    W/m^2 for the past 24 hours in hourly resolution. 
    """

    snow_mass = """
    This is a time series measurement of snow mass in kg/m^3 for the past 24 hours 
    in hourly resolution. 
    """

    snowfall = """
    This is a time series measurement of snow fall in mm/h for the past 24 hours 
    in hourly resolution.  
    """

    temperature_celsius = """
    This is a time series measurement of temperature in degrees Celsius for the 
    past 24 hours in hourly resolution.  
    """

    wind_speed = """
    This is a time series measurement of wind speed in km/h for the past 24 hours 
    in hourly resolution.  
    """

    historic_load = """
    This is a time series of historic electric load in 15-minute resolution for
    the given building at the given time.
    """

    future_load = """
    This is a time series of future electric load in 15-minute resolution for
    the given building at the given time. 
    """

    zoom1 = """
    This is the lowest of three zoom levels.
    """

    zoom2 = """
    This is the intermediate of three zoom levels.
    """

    zoom3 = """
    This is the highest of three zoom levels.
    """

    data_description = {
        'timestamp': timestamp,
        'building_images': building_images,
        'cluster_images_aspect': cluster_images_aspect,
        'cluster_images_ortho': cluster_images_ortho,
        'cluster_images_relief': cluster_images_relief,
        'cluster_images_slope': cluster_images_slope,
        'zoom1': zoom1,
        'zoom2': zoom2,
        'zoom3': zoom3,
        'air_density': air_density,
        'cloud_cover': cloud_cover,
        'precipitation': precipitation,
        'radiation_surface': radiation_surface,
        'radiation_toa': radiation_toa,
        'snow_mass': snow_mass,
        'snowfall': snowfall,
        'temperature_celsius': temperature_celsius,
        'wind_speed': wind_speed,
        'historic_load': historic_load,
        'future_load': future_load
    }

    return data_description