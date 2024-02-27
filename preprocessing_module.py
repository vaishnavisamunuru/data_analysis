from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

class HeartDataPreprocessor:
    def __init__(self, data,numerical_columns=None,categorical_columns=None):
        self.data = data
        self.numerical_columns = numerical_columns if numerical_columns else list(data.select_dtypes(include='number').columns)
        self.categorical_columns = categorical_columns if categorical_columns else list(data.select_dtypes(include='category').columns)

    def handle_missing_values(self):
        if self.data is not None:
            # Replace missing values with the mean for numerical columns
            mean_imputer = SimpleImputer(strategy='mean')
            if self.numerical_columns:
                self.data[self.numerical_columns] = mean_imputer.fit_transform(self.data[self.numerical_columns])
            mode_imputer = SimpleImputer(strategy='most_frequent')
            if self.categorical_columns:
                self.data[self.categorical_columns] = mode_imputer.fit_transform(self.data[self.categorical_columns])

    def encode_categorical_variables(self):
        if self.data is not None:
            # Encode categorical variables using Label Encoding
            if self.categorical_columns:
                label_encoder = LabelEncoder()
                for col in self.categorical_columns:
                    self.data[col] = label_encoder.fit_transform(self.data[col])

    def scale_numerical_features(self,scale_columns):
        if self.data is not None:
            # Standardize numerical features    
            scaler = MinMaxScaler()
            self.data[scale_columns] = scaler.fit_transform(self.data[scale_columns])
