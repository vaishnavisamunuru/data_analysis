import pandas as pd

class HeartDataLoader:
    def __init__(self, file_path):
        self.__file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.__file_path)
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset: {e}")