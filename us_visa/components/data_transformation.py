import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Initializes the DataTransformation class.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads data from a given file path and returns a DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for preprocessing.
        """
        logging.info("Creating data transformer object")

        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore')
            ordinal_encoder = OrdinalEncoder()

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            transform_pipe = Pipeline([
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            
            preprocessor = ColumnTransformer([
                ("OneHotEncoder", oh_transformer, oh_columns),
                ("Ordinal_Encoder", ordinal_encoder, or_columns),
                ("Transformer", transform_pipe, transform_columns),
                ("StandardScaler", numeric_transformer, num_features)
            ])

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates data transformation by preprocessing training and testing datasets.
        """
        try:
            logging.info("Starting data transformation process")

            # Get the preprocessor
            preprocessor = self.get_data_transformer_object()
            logging.info("Obtained preprocessor object")

            # Read data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Separated input and target features")

            # Add derived feature
            input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
            input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
            logging.info("Added company_age column")

            # Drop unwanted columns
            drop_cols = self._schema_config['drop_columns']
            input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
            input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
            logging.info("Dropped specified columns")

            # Convert target variable using mapping
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())
            logging.info("Mapped target variable values")

            # Apply preprocessing
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Applied preprocessing to datasets")

            # Apply SMOTEENN only on training data
            logging.info("Applying SMOTEENN on Training dataset")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            logging.info("Applied SMOTEENN on training dataset")

            # **Skip SMOTEENN on test data to prevent data leakage**
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df

            # Combine input and target features
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Saved transformation artifacts")

            # Return transformation artifact
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise USvisaException(e, sys) from e


