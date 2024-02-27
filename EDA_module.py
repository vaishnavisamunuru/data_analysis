import matplotlib.pyplot as plt
import seaborn as sns
from loading_module import HeartDataLoader

import matplotlib.pyplot as plt
import seaborn as sns

# Inherting HeartDataLoader
class HeartDataEDA(HeartDataLoader):
    def __init__(self, file):
        super().__init__(file)
        self.load_data()

    def plot_correlation_heatmap(self,cols=None):
        if self.data is not None:
            if not cols:
                cols = self.data.select_dtypes(include='number').columns
            corr_matrix = self.data[cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")

    def plot_distribution(self, column):
        if self.data is not None:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")

    def descriptive_stats(self):
        if self.data is not None:
            desc_stats = self.data.describe()
            print("Descriptive Statistics:")
            print(desc_stats)

            # Additional statistics like skewness and kurtosis
            skewness = self.data.skew()
            kurtosis = self.data.kurt()
            print("\nSkewness:")
            print(skewness)
            print("\nKurtosis:")
            print(kurtosis)

            return desc_stats
        else:
            print("Please provide data for EDA.")

    def variable_frequencies(self, column):
        if self.data is not None:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=column, data=self.data)
            plt.title(f'Frequency of {column}')
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")

    def grouped_bar_plot(self, x_column, y_column, hue_column=None):
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            sns.barplot(x=x_column, y=y_column, hue=hue_column, data=self.data)
            plt.title(f'Grouped Bar Plot: {y_column} vs {x_column} (Grouped by {hue_column})')
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")

    def pie_chart(self, column):
        if self.data is not None:
            plt.figure(figsize=(8, 8))
            self.data[column].value_counts().plot.pie(autopct='%1.1f%%')
            plt.title(f'Pie Chart: {column}')
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")

    def box_plot(self, x_column, y_column,hue_column=None):
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            sns.boxplot(x=x_column, y=y_column,hue=hue_column, data=self.data)
            plt.title(f'Box Plot: {y_column} vs {x_column}')
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")

    def scatter_plot(self, x_column, y_column,hue_column=None):
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=x_column, y=y_column,hue=hue_column,data=self.data)
            plt.title(f'Scatter Plot: {y_column} vs {x_column}')
            plt.show()
            return plt
        else:
            print("Please provide data for EDA.")