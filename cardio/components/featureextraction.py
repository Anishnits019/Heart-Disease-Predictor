from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from cardio.entity.artifact_entity import DataValidationArtifact, FeatureExtractionArtifact 
from cardio.entity.config_entity import FeatureExtractionConfig
from cardio.exception.exception import CustomException
from cardio.logging.logger import logging
import pandas as pd
import numpy as np
import os
import sys


class FeatureExtraction:
    def __init__(self, feature_extraction_config: FeatureExtractionConfig, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info("Initializing Feature Extraction Component.")
            self.data_validation_artifact = data_validation_artifact
            self.feature_extraction_config = feature_extraction_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def add_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Adding health and lifestyle features.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            for df in [train_df, test_df]:
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
                df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
                df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
                
                df["Hypertension"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
                df["Age_Cholesterol"] = df["age"] * df["cholesterol"]
                df["Normal_BP"] = ((df["ap_hi"] < 120) & (df["ap_lo"] < 80)).astype(int)
                df["High_Cholesterol"] = (df["cholesterol"] > 1).astype(int)
                df["High_Glucose"] = (df["gluc"] > 1).astype(int)

                bins = [29, 40, 50, 60, 70]
                labels = ['30-40', '40-50', '50-64', '60+']
                df['age_bins'] = pd.cut(df['age'], bins=bins, labels=labels)

                bmi_bins = [0, 18.5, 25, 30, 35, 100]
                bmi_labels = ['underweight', 'normal', 'overweight', 'obese', 'severely_obese']
                df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)

                df['unhealthy_life_style'] = df['smoke'] + df['alco'] + (1 - df['active'])

            logging.info("Health and lifestyle indicators added successfully.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while adding health features.")
            raise CustomException(e, sys)

    @staticmethod
    def bp_category(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Adding BP category features.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            conditions_train = [
                (train_df['ap_hi'] < 120) & (train_df['ap_lo'] < 80),
                (train_df['ap_hi'] < 130),
                (train_df['ap_hi'] < 140)
            ]
            conditions_test = [
                (test_df['ap_hi'] < 120) & (test_df['ap_lo'] < 80),
                (test_df['ap_hi'] < 130),
                (test_df['ap_hi'] < 140)
            ]
            choices = ['normal', 'elevated', 'stage1']

            train_df['bp_category'] = np.select(conditions_train, choices, default='stage2')
            test_df['bp_category'] = np.select(conditions_test, choices, default='stage2')

            logging.info("BP category feature processing completed.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while assigning BP categories.")
            raise CustomException(e, sys)

    def initiate_feature_extraction(self) -> FeatureExtractionArtifact:
        try:
            logging.info("==================== Starting Feature Extraction Pipeline ====================")
            
            logging.info("Loading validated dataset targets from validation artifact paths.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            # Cascade feature engineering pipelines
            train_df, test_df = self.add_features(train_df, test_df)
            train_df, test_df = self.bp_category(train_df, test_df)

            train_file_path = self.feature_extraction_config.training_file_path
            logging.info(f"Persisting processed engineering train matrix split out to: {train_file_path}")
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_df.to_csv(train_file_path, index=False, header=True)

            test_file_path = self.feature_extraction_config.testing_file_path
            logging.info(f"Persisting processed engineering test matrix split out to: {test_file_path}")
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            test_df.to_csv(test_file_path, index=False, header=True)

            logging.info("Constructing complete Feature Extraction Pipeline Output Artifact object.")
            feature_extraction_artifact = FeatureExtractionArtifact(
                trained_file_path=self.feature_extraction_config.training_file_path,
                test_file_path=self.feature_extraction_config.testing_file_path
            )

            logging.info("==================== Feature Extraction Pipeline Success ====================")
            return feature_extraction_artifact 
        except Exception as e:
            logging.error("Critical failure during complete execution of feature extraction pipeline cascades.")
            raise CustomException(e, sys)