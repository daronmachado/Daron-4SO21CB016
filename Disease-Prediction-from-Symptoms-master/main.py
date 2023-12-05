import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn

class DiseasePrediction:
    def __init__(self, model_name=None):
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        self.verbose = self.config['verbose']
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        self.model_name = model_name
        self.model_save_path = self.config['model_save_path']

    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        assert len(train_features.iloc[0]) == 132
        assert len(train_labels) == train_features.shape[0]

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        assert len(test_features.iloc[0]) == 132
        assert len(test_labels) == test_features.shape[0]

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    def _feature_correlation(self, data_frame=None, show_fig=False):
        numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns

        if not numeric_columns.empty:
            corr = data_frame[numeric_columns].corr()
            sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
            plt.title("Feature Correlation")
            plt.tight_layout()
            if show_fig:
                plt.show()
            plt.savefig('feature_correlation.png')
        else:
            print("No numeric columns found for correlation analysis.")

    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        model_name = self.config['model'].get('name', '')
        if model_name != 'decision_tree':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            # Also scale the test features
            self.test_features = scaler.transform(self.test_features)

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    def select_model(self):
        model_name = self.config.get('model', {}).get('name', 'random_forest')

        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [10, 20, 30],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf_classifier = RandomForestClassifier(random_state=self.config['random_state'])
            grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.train_features, self.train_labels)
            print("Best Parameters:", grid_search.best_params_)
            classifier = grid_search.best_estimator_
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        return classifier

    def _train_model(self):
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        classifier = classifier.fit(X_train, y_train)
        confidence = classifier.score(X_val, y_val)
        y_pred = classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        conf_mat = confusion_matrix(y_val, y_pred)
        clf_report = classification_report(y_val, y_pred)
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    def train_model(self):
        try:
            self._train_model()
        except Exception as e:
            print(f"Error during training: {e}")

    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")
            return None, None

        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            try:
                result = clf.predict(self.test_features)
                accuracy = accuracy_score(self.test_labels, result)
                clf_report = classification_report(self.test_labels, result)
                return accuracy, clf_report
            except Exception as e:
                print(f"Error during prediction: {e}")
                return None, None


if __name__ == "__main__":
    current_model_name = 'random_forest'  # Change this to 'decision_tree' if needed
    dp = DiseasePrediction(model_name=current_model_name)
    try:
        dp.train_model()
        test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
        if test_accuracy is not None and classification_report is not None:
            print("Model Test Accuracy: ", test_accuracy)
            print("Test Data Classification Report: \n", classification_report)
    except Exception as e:
        print(f"Error: {e}")

