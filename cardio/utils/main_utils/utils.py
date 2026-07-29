from pymongo import results
import yaml
import os,sys
import numpy as np
import pandas as pd
from cardio.exception.exception import CustomException
from cardio.constant.training_pipeline import SCHEMA_FILE_PATH
from cardio.logging.logger import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import json
def save_data(file_path,data):
    try:
         os.makedirs(os.path.dirname(file_path),exist_ok=True)
         with open(file_path, "a") as f:
          json.dump(data, f, indent=4)
    except Exception as e:
      raise CustomException(e,sys)

def read_yaml_file(file_path):
    with open (file_path,"r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except Exception as e:
            raise CustomException(e,sys)
        
def write_yaml_file(file_path,content):
    try:
     with open (file_path,"w") as file_obj:
        yaml.dump(content,file_obj)
    except Exception as e:
     raise CustomException(e,sys)

def save_numpy_array(file_path,data):
    try:
       os.makedirs(os.path.dirname(file_path),exist_ok=True)
       with open (file_path,"wb") as file_obj:
        np.save(file_obj,data)
    except Exception as e:
      raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
       os.makedirs(os.path.dirname(file_path),exist_ok=True)
       with open (file_path,"wb") as file_obj:
        pickle.dump(obj,file_obj)
    except Exception as e:
      raise CustomException(e,sys)

def load_numpy_array(file_path):
    try:
         with open (file_path,"rb") as file_obj:
          return np.load(file_obj)
    except Exception as e:
      raise CustomException(e,sys)

def evaluate_models(x_train, x_test, y_train, y_test, models, params, file_path):
    results = {}
    for name, model in models.items():

        gs = GridSearchCV(
            estimator=model,
            param_grid=params[name],
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )
        # n_jobs=-1 set all the used all vritual cores of the system 
        gs.fit(x_train, y_train)

        best_model = gs.best_estimator_

        # Predictions (Labels)
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        # Prediction Probabilities (for ROC-AUC score)
        y_train_prob = None
        y_test_prob = None
        if hasattr(best_model, "predict_proba"):
            y_train_prob = best_model.predict_proba(x_train)[:, 1]
            y_test_prob = best_model.predict_proba(x_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_train_prob = best_model.decision_function(x_train)
            y_test_prob = best_model.decision_function(x_test)

        # 1. Accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # 2. F1 Score
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # 3. Precision
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)

        # 4. Recall
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        # 5. ROC-AUC Score
        train_roc_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else None
        test_roc_auc = roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else None

        # Consolidate scores into dictionaries
        train_scores = {
            "accuracy": train_accuracy,
            "f1_score": train_f1,
            "precision": train_precision,
            "recall": train_recall,
            "roc_auc": train_roc_auc
        }

        test_scores = {
            "accuracy": test_accuracy,
            "f1_score": test_f1,
            "precision": test_precision,
            "recall": test_recall,
            "roc_auc": test_roc_auc
        }

        results[name] = {
            "train_score": train_scores,
            "test_score": test_scores,
            "best_model": best_model,
        }

        json_result = {
            "time": str(pd.Timestamp.now()),
            "train_score": train_scores,
            "test_score": test_scores,
            "best_model_name": type(best_model).__name__,
            "best_model_params": gs.best_params_,
        }
        save_data(file_path, json_result)

        # Terminal output for monitoring progress
        border = "=" * 50
        print(f"\n{border}")
        print(f"📊 MODEL: {name.upper()}")
        print(f"{border}")
        print("  [TRAIN METRICS]")
        print(f"   • Accuracy  : {train_accuracy:.4f}")
        print(f"   • F1 Score  : {train_f1:.4f}")
        print(f"   • Precision : {train_precision:.4f}")
        print(f"   • Recall    : {train_recall:.4f}")
        if train_roc_auc is not None:
            print(f"   • ROC-AUC   : {train_roc_auc:.4f}")
            
        print("  [TEST METRICS]")
        print(f"   • Accuracy  : {test_accuracy:.4f}")
        print(f"   • F1 Score  : {test_f1:.4f}")
        print(f"   • Precision : {test_precision:.4f}")
        print(f"   • Recall    : {test_recall:.4f}")
        if test_roc_auc is not None:
            print(f"   • ROC-AUC   : {test_roc_auc:.4f}")
        print(f"{border}\n", flush=True)

    return results