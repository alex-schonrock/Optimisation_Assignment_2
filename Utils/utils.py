import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
### load data file function
def load_datafile(file_name, input_path):
    base_path = Path(input_path) / file_name
    result = pd.read_csv(base_path, sep=";")
    # print(result)
    return result


### plotting functions