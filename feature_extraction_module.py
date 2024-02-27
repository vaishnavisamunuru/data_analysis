from loading_module import HeartDataLoader
from preprocessing_module import HeartDataPreprocessor
from sklearn.decomposition import PCA
import pandas as pd

class HeartFeatureExtractor:
    def __init__(self, data,numerical_cols,target_column):
        self.data = data
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        self.numerical_cols = numerical_cols

    def apply_pca(self, n_components=None):
        # Example PCA application (customize based on your needs)
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(self.data[self.numerical_cols])
        transformed_df = pd.DataFrame(transformed_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])
        self.data = pd.concat([self.data, transformed_df], axis=1)