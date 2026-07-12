import os
import pandas as pd

def test_data_exists():
    assert os.path.exists("data/heart_clean.csv")
