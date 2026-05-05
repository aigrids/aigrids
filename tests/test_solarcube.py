"""Tests loading of standardized tasks for SolarCube.


Example usage:
--------------
    $ python tests/test_solarcube.py

or
    $ pytest tests/test_solarcube.py

"""
import os
import string
import random

from aidotgrids import load

# create a random root path name
random_string = ''.join(
    random.choice(string.ascii_letters) for _ in range(7)
)

root_path = os.path.expanduser(
    "~/AI-grids/test_solarcube_" + random_string
)

dataset = load.load_task(
    "SolarCube", 
    "odd_time_area", 
    root_path,
    data_frac = 0.01,
    train_frac = 0.1
)

print("Successfully executed 'test_solarcube.py'!")