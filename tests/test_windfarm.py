"""Tests loading of standardized tasks for WindFarm.


Example usage:
--------------
    $ python tests/test_windfarm.py

or
    $ pytest

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
    "~/AI-grids/test_windfarm_" + random_string
)

dataset = load.load_task(
    "WindFarm", 
    "odd_time_predict48h", 
    root_path,
    data_frac = 0.01,
    train_frac = 0.1
)

print("Successfully executed 'test_windfarm.py'!")