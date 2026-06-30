"""General utility functions for loading data of all tasks."""
import random 

def shuffle_datadict(dataset: dict, seed: int) -> dict:
    """Shuffle a dictionary by key."""
    items = list(dataset.items())
    random.Random(seed).shuffle(items)
    return dict(items)