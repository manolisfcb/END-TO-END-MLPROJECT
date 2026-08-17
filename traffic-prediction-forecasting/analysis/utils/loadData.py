from abc import ABC, abstractmethod
import numpy as np

class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass
    

class CSVDataLoader(DataLoader):
    def __init__(self, file_path, separator=','):
        self.file_path = file_path
        self.separator = separator

    def load_data(self):
        import pandas as pd
        return pd.read_csv(self.file_path, sep=self.separator)
    

class JSONDataLoader(DataLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        import pandas as pd
        return pd.read_json(self.file_path)


class NPYDataLoader(DataLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        arr = np.load(self.file_path, mmap_mode='r')
        sensor_data = arr[:,:,:]
        return sensor_data