"""Tests loading of standardized tasks for SystemImbalance.


Example usage:
--------------
    $ python tests/test_systemimbalance.py

or
    $ pytest tests/test_systemimbalance.py

"""
import os
import string
import random

from aigrids import load

# create a random root path name
random_string = ''.join(
    random.choice(string.ascii_letters) for _ in range(7)
)

root_path = os.path.expanduser(
    "~/AI-grids/test_systemimbalance_" + random_string
)

dataset = load.load_task(
    "SystemImbalance",
    "odd_time_volume_predict1h",
    root_path,
    data_frac = 0.01,
    train_frac = 0.1
)

print("Successfully executed 'test_systemimbalance.py'!")
