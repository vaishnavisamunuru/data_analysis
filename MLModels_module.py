from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

class HeartClassifier:
    def __init__(self, data, target_column,indept_columns=None):
        self.data = data
        self.target_column = target_column
        self.indept_columns = indept_columns

    def split_data(self,data):
        if not data:
            data=self.data.copy()
        X = data[self.indept_columns]
        y = data[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=40)

    def model_fit(self,model_name='Logistic Regression',data=None):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = self.split_data(data)

        smote = SMOTE(sampling_strategy=1)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        print(f'Results with {self.target_column} as target column')
        print('_______________________________________________')
        # Train and evaluate models using the original data
        print("Results with Data:")
        if model_name == 'Random Forest':
            return self.train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
        elif model_name == 'Logistic Regression':
            return self.train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
        elif model_name == 'SVC':
            return self.train_and_evaluate_support_vector_machine(X_train, X_test, y_train, y_test)
                

    def train_and_evaluate_random_forest(self, X_train, X_test, y_train, y_test):
        pipeline_rf = Pipeline([('classifier', RandomForestClassifier(random_state=42))])

        param_grid_rf = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [5, 10],
            'classifier__min_samples_leaf': [2, 4]
        }

        return self._train_and_evaluate_model(pipeline_rf, param_grid_rf, X_train, X_test, y_train, y_test)

    def train_and_evaluate_logistic_regression(self, X_train, X_test, y_train, y_test):
        pipeline_lr = Pipeline([('classifier', LogisticRegression(random_state=42))])

        param_grid_lr = {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'saga']
        }

        return self._train_and_evaluate_model(pipeline_lr, param_grid_lr, X_train, X_test, y_train, y_test)

    def train_and_evaluate_support_vector_machine(self, X_train, X_test, y_train, y_test):
        pipeline_svc = Pipeline([('classifier', SVC(random_state=42))])

        param_grid_svc = {
            'classifier__C': [0.1, 1],
            'classifier__kernel': ['rbf', 'poly'],
            'classifier__gamma': ['auto']
        }

        return self._train_and_evaluate_model(pipeline_svc, param_grid_svc, X_train, X_test, y_train, y_test)

    def _train_and_evaluate_model(self, pipeline, param_grid, X_train, X_test, y_train, y_test):
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        # Print classification report
        classification_rep_str = classification_report(y_test, y_pred)
        print(f"Classification Report for {best_model.named_steps['classifier'].__class__.__name__}:")
        print(classification_rep_str)

        # Print best hyperparameters
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        return classification_rep_str

    def save_to_file(self, content, filename):
        with open(filename, 'w') as file:
            file.write(content)

    def plot_and_save_confusion_matrix(self, y_test, y_pred, filename):
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix: ')
        print(cm)
        disp = ConfusionMatrixDisplay(cm,
                                     display_labels=['Survived', 'Not Survived'])
        # disp.ax_.set_title('Confusion Matrix')
        disp.plot()
        # plt.show()
        plt.savefig(filename)
        plt.close()