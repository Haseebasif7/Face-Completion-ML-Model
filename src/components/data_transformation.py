import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads data from a CSV file and returns a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process, including splitting, PCA transformation, 
        and saving the transformed data.
        """
        try:
            logging.info("Data Transformation Started !!!")

            # Load train and test data
            train_data = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_data = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Drop the target column 'Y' from both datasets
            train_data = train_data.drop('Y', axis=1)
            test_data = test_data.drop('Y', axis=1)

            # Split the data into X (upper half) and Y (lower half)
            X_train = train_data.iloc[:, :train_data.shape[1] // 2]
            Y_train = train_data.iloc[:, train_data.shape[1] // 2:]
            X_test = test_data.iloc[:, :test_data.shape[1] // 2]
            Y_test = test_data.iloc[:, test_data.shape[1] // 2:]
            logging.info("Data split into upper and lower halves")

            # Apply PCA transformation
            components = 90
            logging.info("Applying PCA transformation with %d components", components)

            pca = PCA(n_components=components, svd_solver='auto')
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            pca1 = PCA(n_components=components, svd_solver='auto')
            Y_train_pca = pca1.fit_transform(Y_train)
            Y_test_pca = pca1.transform(Y_test)
            logging.info("PCA transformation completed")

            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, 
                                  array=np.c_[X_train_pca, Y_train_pca])
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, 
                                  array=np.c_[X_test_pca, Y_test_pca])
            logging.info("Transformed data saved successfully")

            # Model training
            param_grid = {
                "Random Forest": [200, 500],
                "Extra Trees": [200, 500],
            }

            ESTIMATORS = {
                "Random Forest": [
                    MultiOutputRegressor(RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1)) 
                    for n in param_grid["Random Forest"]
                ],
                "Extra Trees": [
                    MultiOutputRegressor(ExtraTreesRegressor(n_estimators=n, random_state=0, n_jobs=-1)) 
                    for n in param_grid["Extra Trees"]
                ],
            }

            results = {}
            for name, models in ESTIMATORS.items():
                print(f"\nTraining {name} models...")
                for model in models:
                    n_estimators = model.estimator.n_estimators
                    print(f"Training {name} with n_estimators={n_estimators}...")

                    model.fit(X_train_pca, Y_train_pca)

                    Y_pred_pca = model.predict(X_test_pca)
                    Y_pred = pca1.inverse_transform(Y_pred_pca)

                    mae = mean_absolute_error(Y_test, Y_pred)

                    results[f"{name} (n_estimators={n_estimators})"] = {
                        "model": model,
                        "predictions": Y_pred,
                        "mae": mae,
                    }
            logging.info("Model training completed")

            n_faces = 5
            n_cols = 1 + len(results)
            image_shape = (64, 64)

            plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))
            plt.suptitle("Face Completion with Multi-Output Estimators", size=16)

            for i in range(n_faces):
                X_upper = pca.inverse_transform(X_test_pca[i])
                Y_lower_true = Y_test.iloc[i].values
                true_face = np.hstack((X_upper, Y_lower_true))

                sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
                if i == 0:
                    sub.set_title("True Faces", fontsize=10)
                sub.axis("off")
                sub.imshow(true_face.reshape(image_shape), cmap="gray", interpolation="nearest")

                for j, (name, result) in enumerate(results.items()):
                    Y_lower_pred = result["predictions"][i]
                    completed_face = np.hstack((X_upper, Y_lower_pred))

                    sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
                    if i == 0:
                        sub.set_title(f"{name}\nMAE: {result['mae']:.2f}", fontsize=8)
                    sub.axis("off")
                    sub.imshow(completed_face.reshape(image_shape), cmap="gray", interpolation="nearest")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
            logging.info("Results Plotted")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
