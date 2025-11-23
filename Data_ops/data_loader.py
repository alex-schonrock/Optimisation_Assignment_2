import pandas as pd
from pathlib import Path

from pathlib import Path
from dataclasses import dataclass
from logging import Logger
import pandas as pd
from Utils.utils import load_datafile
#### data loader class
class DataLoader:
    def __init__(self, input_path: str, model_type: str):
        self.input_path = Path(input_path).resolve()
        self.model_type = model_type
    def load_data_file(self, file_name: str):
        data = load_datafile(file_name, self.input_path)
        return data

