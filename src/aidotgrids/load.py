"""Main module for loading AI.grids tasks in standardized data format.

Example usage:
--------------
    from aidotgrids import load

    ds = load.load_task(
        task_name='WindFarm', 
        subtask_name='odd_time_predict48h',
        root_path='~/AI-grids/'
    )

"""
import os
import sys
import time
import math
import requests
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# individual task loading modules
from . import opfdata
from . import powergraph
from . import solarcube
from . import buildingelectricity
from . import windfarm


# Global variables
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
DOWNLOAD_RETRIES = 20
DOWNLOAD_TIMEOUT = (30, 180)  # connect, read timeout

HUGGING_FACE_BASE = 'AI-grids'


LIST_AVAIL_TASKNAMES = [
    'OPFData',
    'PowerGraph',
    'SolarCube',
    'BuildingElectricity',
    'WindFarm'
]

def load_task(
    task_name: str,
    subtask_name: str,
    root_path: str = '~/AI-grids',
    data_frac: int = 1,
    train_frac: int = 1,
    max_workers: int = 1024,
    max_workers_download: int = 2
) -> dict:
    """
    Download task repository (if needed) and load standardized subtask.
    
    """
    # validate input arguments
    _validate_inputs(
        task_name, 
        subtask_name, 
        root_path, 
        data_frac, 
        train_frac,
        max_workers, 
        max_workers_download
    )

    # replace ~ with full path. safer
    root_path = os.path.expanduser(root_path)

    # expand root_path with taskname
    local_dir = os.path.join(root_path, task_name)

    # download task repository (skips already-downloaded & already-uncompressed)
    _download_hf_repo(
        local_dir, 
        task_name,
        max_workers,
        max_workers_download
    )

    # load subtask
    subtask_data = _load_subtask(
        local_dir, 
        subtask_name, 
        data_frac,
        train_frac,
        max_workers
    )
    
    return subtask_data


def _load_subtask(
    local_dir: str, 
    subtask_name: str,
    data_frac: int,
    train_frac: int,
    max_workers: int
) -> dict:
    """
    Load standardized task.
    
    """
    print(f"Preparing subtask {subtask_name}")
    if 'OPFData' in local_dir:
        subtask_data = opfdata.load(
            local_dir, 
            subtask_name,
            data_frac,
            train_frac,
            max_workers
        )
    elif 'PowerGraph' in local_dir:
        subtask_data = powergraph.load(
            local_dir,
            subtask_name,
            data_frac,
            train_frac,
            max_workers
        )
    elif 'SolarCube' in local_dir:
        subtask_data = solarcube.load(
            local_dir,
            subtask_name,
            data_frac,
            train_frac,
            max_workers
        )
    elif 'BuildingElectricity' in local_dir:
        subtask_data = buildingelectricity.load(
            local_dir,
            subtask_name,
            data_frac,
            train_frac,
            max_workers
        )
    elif 'WindFarm' in local_dir:
        subtask_data = windfarm.load(
            local_dir,
            subtask_name,
            data_frac,
            train_frac,
            max_workers
        )
    else:
        raise NotImplementedError("Loading of selected task not implemented!")

    print(f"Data for {subtask_name} successfully loaded.\n")
    
    return subtask_data


def _download_hf_repo(
    local_dir: str, 
    task_name: str, 
    max_workers: int,
    max_workers_download: int
):
    """
    Download and uncompress all files from Hugging Face in parallel, skipping 
    download where files or final uncompressed data already exist, and skipping 
    decompression if data is already extracted (but removing the compressed file 
    if present).

    """
    print(f"Preparing local data directory for {task_name}")

    repo_id = f'{HUGGING_FACE_BASE}/{task_name}'

    # Step 1: Collect all files (recursively)
    files_to_download = []
    _collect_files(repo_id, local_dir, subpath="", files_list=files_to_download)

    # Step 2: Download them in parallel (only if needed)
    print(f"\nFound {len(files_to_download)} files to manage (download if needed).")
    with ThreadPoolExecutor(max_workers=max_workers_download) as executor:
        future_to_file = {}
        for (url, local_path) in files_to_download:
            submit_job = executor.submit(_download_single_file, url, local_path)
            future_to_file[submit_job] = (url, local_path)

        for future in as_completed(future_to_file):
            url, local_path = future_to_file[future]
            try:
                future.result()  # raises if download fails

            except Exception as e:
                raise RuntimeError(f"Download failed for {url}") from e

    print(f"Data for {repo_id} is now available at {local_dir}.\n")

    # Step 3: Uncompress files in parallel if needed
    compressed_exts = (".zip", ".tar.gz", ".tar")
    compressed_files = []
    for (_, local_path) in files_to_download:
        if local_path.endswith(compressed_exts) and os.path.exists(local_path):
            compressed_files.append(local_path)

    if compressed_files:
        print(f"Uncompressing {len(compressed_files)} files (if needed) in parallel.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_compressed = {}
            for path in compressed_files:
                submit_job = executor.submit(_uncompress_and_delete_file, path)
                future_to_compressed[submit_job] = path

            for future in as_completed(future_to_compressed):
                path = future_to_compressed[future]
                try:
                    future.result()
                    
                except Exception as e:
                    print(f"Uncompress & delete failed for {path}: {e}")
        
        print("All compressed files have been handled.")


def _collect_files(repo_id: str, local_dir: str, subpath: str, files_list: list):
    """
    Recursively gather file paths from the Hugging Face API 
    and append them as (file_url, local_entry_path) tuples to files_list.

    """
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    if subpath:
        api_url += f"/{subpath}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list: {e}")
        return

    content_list = response.json()
    
    for entry in content_list:
        entry_type = entry['type']
        entry_path = entry['path']
        local_entry_path = os.path.join(local_dir, entry_path)

        if entry_type == 'file':
            file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{entry_path}"
            files_list.append((file_url, local_entry_path))

        elif entry_type == 'directory':
            _collect_files(repo_id, local_dir, subpath=entry_path, files_list=files_list)

def _download_single_file(url: str, local_path: str):
    """
    Download a single file with resume support.

    Writes to local_path + ".part" first, resumes incomplete downloads via HTTP
    Range requests, and only moves to the final path after the expected size is
    reached.
    """

    if os.path.exists(local_path):
        print(f"[SKIP] {local_path} already present.")
        return

    if _is_compressed_file(local_path):
        base_folder = _guess_uncompressed_dir(local_path)
        if base_folder and os.path.isdir(base_folder):
            print(f"[SKIP] '{base_folder}' is already uncompressed.")
            return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    part_path = local_path + ".part"
    session = _make_session()
    expected_size = _remote_file_size(url, session)

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        existing_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0

        if expected_size is not None and existing_size == expected_size:
            os.replace(part_path, local_path)
            print(f"[DONE] {local_path}")
            return

        headers = {}
        mode = "wb"

        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            print(
                f"Downloading {url} -> {local_path} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES}, resume at {existing_size} bytes)"
            )

            with session.get(
                url,
                stream=True,
                headers=headers,
                timeout=DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            ) as response:

                # If resume was requested but server ignores Range, restart cleanly.
                if existing_size > 0 and response.status_code != 206:
                    print("[WARN] Server did not honor Range request. Restarting file.")
                    os.remove(part_path)
                    continue

                response.raise_for_status()

                with open(part_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            final_size = os.path.getsize(part_path)

            if expected_size is not None and final_size != expected_size:
                raise IOError(
                    f"incomplete download: got {final_size} bytes, "
                    f"expected {expected_size} bytes"
                )

            os.replace(part_path, local_path)
            print(f"[DONE] {local_path}")
            return

        except Exception as e:
            print(f"[WARN] Download interrupted for {url}: {e}")
            time.sleep(min(120, 2 ** attempt))

    raise RuntimeError(f"Failed to download after {DOWNLOAD_RETRIES} attempts: {url}")


def _uncompress_and_delete_file(file_path: str):
    """
    Decompress a zip/tar/tar.gz file using Python standard libraries, then delete
    the compressed file. If the uncompressed data is already present, skip
    decompression but still remove the compressed file.
    """
    if not os.path.exists(file_path):
        return

    parent_dir = os.path.dirname(file_path)
    base_folder = _guess_uncompressed_dir(file_path)

    if base_folder and os.path.isdir(base_folder):
        print(f"[SKIP] already decompressed: {base_folder}. Remove {file_path}.")
        os.remove(file_path)
        return

    print(f"Uncompressing {file_path}.")

    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as archive:
            archive.extractall(parent_dir)

    elif file_path.endswith(".tar.gz") or file_path.endswith(".tar"):
        with tarfile.open(file_path, "r:*") as archive:
            archive.extractall(parent_dir, filter="data")

    else:
        print(f"[SKIP] Unrecognized compression format: {file_path}")
        return

    print(f"Deleting {file_path}.")
    os.remove(file_path)

def _is_compressed_file(local_path: str) -> bool:
    """
    Return True if the path ends with a known compression extension.
    
    """
    compressed_exts = (".zip", ".tar.gz", ".tar")
    return any(local_path.endswith(ext) for ext in compressed_exts)


def _guess_uncompressed_dir(compressed_path: str) -> str:
    """
    Given a compressed file name, guess the resulting directory name
    after decompression. This is a naive approach:
      - .zip    -> strip .zip
      - .tar.gz -> strip .tar.gz
      - .tar    -> strip .tar

    """
    # If .tar.gz
    if compressed_path.endswith(".tar.gz"):
        return compressed_path[:-7]  # remove .tar.gz
    # If .zip
    if compressed_path.endswith(".zip"):
        return compressed_path[:-4]  # remove .zip
    # If .tar
    if compressed_path.endswith(".tar"):
        return compressed_path[:-4]  # remove .tar
    return ""  # no guess if unrecognized


def _validate_inputs(
    task_name: str,
    subtask_name: str,
    root_path: str,
    data_frac: float,
    train_frac: int,
    max_workers: int,
    max_workers_download: int
):
    """
    Check user inputs and correct if necessary.
    
    """
    # validate task_name
    if task_name not in LIST_AVAIL_TASKNAMES:
        sys.exit(f"Selected task '{task_name}' not recognized.")

    else:
        print(f"Processing task {task_name} for subtask {subtask_name}.")

    # validate root_path
    if type(root_path) is not str:
        print("root_path defined by user is not string. Set to '~/AI-grids'")
        root_path = '~/AI-grids'

    # validate data_frac
    if data_frac <= 0 or data_frac > 1:
        print(f'The data_frac is set to {data_frac}. This is invalid. Resuming',
            'with a value of data_frac=1.')
        data_frac = 1

    else:
        print(f'Using {data_frac:.0%} of total data as specified by user.')

    # validate max_workers
    if (
        type(max_workers) is not int
        and type(max_workers) is not float
    ):
        print(f'Invalid argument {max_workers} for max_workers. Setting',
            'max_workers=1024')
        max_workers = 1024

    else:
        # setting ceiled value for max_workers for the case of passed fraction.
        max_workers = math.ceil(max_workers)
        print(f'Continuing with max_workers={max_workers}')

    # validate max_workers_download
    if (
        type(max_workers_download) is not int
        and type(max_workers_download) is not float
    ):
        print(f'Invalid argument {max_workers_download} for max_workers_download.',
            'Setting max_workers_download=4.')
        max_workers_download = 4

    else:
        # ceil value for max_workers_download in the case of passed fraction.
        max_workers_download = math.ceil(max_workers_download)
        print(f'Continuing with max_workers_download={max_workers_download}')


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _remote_file_size(url: str, session: requests.Session) -> int | None:
    r = session.head(url, allow_redirects=True, timeout=DOWNLOAD_TIMEOUT)
    if r.status_code >= 400:
        return None
    size = r.headers.get("Content-Length")
    return int(size) if size is not None else None