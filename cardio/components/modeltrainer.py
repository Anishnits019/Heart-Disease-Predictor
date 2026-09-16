from cardio.entity.config_entity import ModelTrainerConfig
from cardio.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from cardio.exception.exception import CustomException
from cardio.utils.main_utils.utils import save_object, load_numpy_array
from cardio.utils.ml_utils.metric.classification_metric import calculate_metrics

import os
import pandas as pd
import sys
import optuna
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduces console clutter

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def track_mlflow(self, best_model, classification_metric, tag, model_name):
        with mlflow.start_run():
            mlflow.set_tag("dataset", tag)
            mlflow.set_tag("model", model_name)

            mlflow.log_metrics({
                "f1_score": classification_metric.f1_score,
                "recall_score": classification_metric.recall_score,
                "precision_score": classification_metric.precision_score
            })
            
            mlflow.sklearn.log_model(best_model, "model")

    def objective(self, trial, model_name, x_train, y_train, x_test, y_test):
        """Dynamic objective function for Optuna tuning across all model types."""
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
                splitter=trial.suggest_categorical("splitter", ["best", "random"]),
                max_features=trial.suggest_categorical("max_features", ["sqrt", 2, 4]),
                random_state=42
            )
        elif model_name == "Random Forest":
            model = RandomForestClassifier(
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
                max_features=trial.suggest_categorical("max_features", ["sqrt", 2, 4]),
                n_estimators=trial.suggest_categorical("n_estimators", [200, 300]),
                random_state=42
            )
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(
                loss=trial.suggest_categorical("loss", ["log_loss", "exponential"]),
                learning_rate=trial.suggest_categorical("learning_rate", [0.05, 0.01]),
                subsample=trial.suggest_categorical("subsample", [0.7, 0.8]),
                criterion=trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
                n_estimators=trial.suggest_categorical("n_estimators", [50, 100, 200, 300]),
                max_depth=trial.suggest_int("max_depth", 3, 5),
                random_state=42
            )
        elif model_name == "AdaBoost":
            model = AdaBoostClassifier(
                learning_rate=trial.suggest_categorical("learning_rate", [0.1, 0.05, 0.01]),
                n_estimators=trial.suggest_categorical("n_estimators", [50, 100, 200, 300]),
                random_state=42
            )
        elif model_name == "Logistic Regression":
            model = LogisticRegression(
                penalty=trial.suggest_categorical("penalty", ["l2"]),
                C=trial.suggest_categorical("C", [0.01, 0.1, 1.0, 10.0]),
                solver=trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                max_iter=trial.suggest_categorical("max_iter", [100, 500]),
                random_state=42
            )
        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                iterations=trial.suggest_categorical("iterations", [400]),
                learning_rate=trial.suggest_categorical("learning_rate", [0.05]),
                depth=trial.suggest_categorical("depth", [6]),
                l2_leaf_reg=trial.suggest_categorical("l2_leaf_reg", [3]),
                bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bernoulli"]),
                subsample=trial.suggest_categorical("subsample", [0.8]),
                random_state=42,
                verbose=0
            )
        elif model_name == "Hist Gradient Boosting":
            model = HistGradientBoostingClassifier(
                learning_rate=trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
                max_iter=trial.suggest_categorical("max_iter", [100, 200]),
                max_depth=trial.suggest_categorical("max_depth", [3, 5, 7]),
                random_state=42
            )
        elif model_name == "XGBoost":
            model = XGBClassifier(
                learning_rate=trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.1]),
                n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300]),
                max_depth=trial.suggest_int("max_depth", 3, 6),
                subsample=trial.suggest_categorical("subsample", [0.8, 0.9]),
                colsample_bytree=trial.suggest_categorical("colsample_bytree", [0.8, 1.0]),
                gamma=trial.suggest_categorical("gamma", [0, 0.1, 0.2]),
                eval_metric='logloss',
                random_state=42
            )
        elif model_name == "LightGBM":
            model = LGBMClassifier(
                learning_rate=trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
                n_estimators=trial.suggest_categorical("n_estimators", [100, 200]),
                max_depth=trial.suggest_categorical("max_depth", [3, 5, 7]),
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        score = f1_score(y_test, preds, average='weighted')
        return score

    def train_model(self, x_train, x_test, y_train, y_test, file_path):
        models_to_tune = [
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "AdaBoost",
            "Logistic Regression",
            "CatBoost",
            "Hist Gradient Boosting",
            "XGBoost",
            "LightGBM"
        ]

        best_overall_score = -1.0
        best_overall_model = None
        best_model_name = ""
        best_hyperparameters = {}

        try:
            print("\n[INFO] Starting Optuna Hyperparameter Tuning across models...")
            
            # Loop through each model and find its optimal hyperparams using Optuna
            for model_name in models_to_tune:
                print(f"[INFO] Tuning {model_name}...")
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: self.objective(trial, model_name, x_train, y_train, x_test, y_test),
                    n_trials=10  # Adjust trial count based on your performance needs
                )
                
                print(f" > Best score for {model_name}: {study.best_value:.4f}")
                
                if study.best_value > best_overall_score:
                    best_overall_score = study.best_value
                    best_model_name = model_name
                    best_hyperparameters = study.best_params

            print(f"\n[INFO] Best Overall Model: {best_model_name} with F1-Score: {best_overall_score:.4f}")

            # Re-instantiate and fit the absolute best model with its optimal params
            if best_model_name == "Decision Tree":
                best_overall_model = DecisionTreeClassifier(**best_hyperparameters, random_state=42)
            elif best_model_name == "Random Forest":
                best_overall_model = RandomForestClassifier(**best_hyperparameters, random_state=42)
            elif best_model_name == "Gradient Boosting":
                best_overall_model = GradientBoostingClassifier(**best_hyperparameters, random_state=42)
            elif best_model_name == "AdaBoost":
                best_overall_model = AdaBoostClassifier(**best_hyperparameters, random_state=42)
            elif best_model_name == "Logistic Regression":
                best_overall_model = LogisticRegression(**best_hyperparameters, random_state=42)
            elif best_model_name == "CatBoost":
                best_overall_model = CatBoostClassifier(**best_hyperparameters, random_state=42, verbose=0)
            elif best_model_name == "Hist Gradient Boosting":
                best_overall_model = HistGradientBoostingClassifier(**best_hyperparameters, random_state=42)
            elif best_model_name == "XGBoost":
                best_overall_model = XGBClassifier(**best_hyperparameters, eval_metric='logloss', random_state=42)
            elif best_model_name == "LightGBM":
                best_overall_model = LGBMClassifier(**best_hyperparameters, random_state=42, verbose=-1)

            best_overall_model.fit(x_train, y_train)

            # Calculate and log metrics
            y_train_pred = best_overall_model.predict(x_train)
            classification_train_metric = calculate_metrics(y_train, y_train_pred)
            self.track_mlflow(best_overall_model, classification_train_metric, "train", best_model_name)

            y_test_pred = best_overall_model.predict(x_test)
            classification_test_metric = calculate_metrics(y_test, y_test_pred)
            self.track_mlflow(best_overall_model, classification_test_metric, "test", best_model_name)

            # Save model artifact
            save_object(self.model_trainer_config.trained_model_file_path, best_overall_model)
       
            model_trainer_artifact = ModelTrainerArtifact(
                model_obj_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
     
    def initiate_model_training(self):
        try:
            train_transformed_data = load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_transformed_data = load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = (
                train_transformed_data[:, :-1],
                train_transformed_data[:, -1],
                test_transformed_data[:, :-1],
                test_transformed_data[:, -1]
            )
            
            model_train_artifact = self.train_model(
                x_train, x_test, y_train, y_test, 
                self.model_trainer_config.model_evaluation_dir
            )
            return model_train_artifact
            
        except Exception as e:
            raise CustomException(e, sys)