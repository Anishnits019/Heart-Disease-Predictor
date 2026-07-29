from cardio.entity.config_entity import ModelTrainerConfig
from cardio.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from cardio.exception.exception import CustomException
from cardio.utils.main_utils.utils  import evaluate_models,save_object,load_numpy_array
from cardio.utils.ml_utils.metric.classification_metric import calculate_metrics

import os
import pandas as pd
import sys

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import (
#     AdaBoostClassifier,
#     GradientBoostingClassifier,
#     RandomForestClassifier,
# )
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
          self.model_trainer_config=model_trainer_config
          self.data_transformation_artifact=data_transformation_artifact

    def track_mlflow(self,best_model,classfication_metric,tag):
        with mlflow.start_run():
             
            mlflow.set_tag("dataset",tag)
            mlflow.set_tag("model",type(best_model).__name__)

            mlflow.log_metrics({
                 "f1_score":classfication_metric.f1_score,
                 "recall_score":classfication_metric.recall_score,
                 "precision_score":classfication_metric.precision_score
             })
            
            mlflow.sklearn.log_model(best_model,"model")

    def train_model(self,x_train,x_test,y_train,y_test,file_path):
         
        models = {
                # "Random Forest": RandomForestClassifier(verbose=1),
                # "Decision Tree": DecisionTreeClassifier(),
                # "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                # "Logistic Regression": LogisticRegression(verbose=1),
                # "AdaBoost": AdaBoostClassifier(),
                # "Hist Gradient Boosting": HistGradientBoostingClassifier(),
                # "XGBoost": XGBClassifier(eval_metric='logloss'),
                # "LightGBM": LGBMClassifier()
                "CatBoost":CatBoostClassifier(verbose=1)
            }
        params={
            # "Decision Tree": {
            #     'max_depth':[3,4,5],               # allowed a definite depth such that model doesn't memeorize till infinite depth
            #     'criterion':['gini', 'entropy'],   # criterion for creating a tree
            #     'min_samples_split':[5,10,20],     # min samples requried to split a tree
            #     'min_samples_leaf':[5,10,20],      # min samples required or left on leaf to split
            #     'splitter':['best','random'],      # best parameters selection or random parameter selection
            #     'max_features':['sqrt',2,4],       # how many features selectres based on sqrt(n) OR LOG(n) where N is the no of columns in a database
            # },
            # "Random Forest":{
            #     'criterion': ['gini', 'entropy'],  # criterion for creating a tree
            #     'max_depth':[4,6,8,10],        # allowed a definite depth such that model doesn't memeorize till infinite depth
            #     'min_samples_split':[5,10,20],   # min samples requried to split a tree
            #     'min_samples_leaf':[5,10,20],    # min samples required or left on leaf to split
            #     'max_features':['sqrt',2,4],  # how many features selectres based on sqrt(n) OR LOG(n) where N is the no of columns in a database
            #     'n_estimators': [200, 300], # independent tress to create for judgement           
            # },
            # "Gradient Boosting":{
            #     'loss':['log_loss', 'exponential'],
            #     'learning_rate':[.05,.01],
            #     'subsample':[0.7,0.8],
            #     'criterion':['squared_error', 'friedman_mse'],
            #     'n_estimators': [50, 100, 200, 300],             # independent tress to create for judgement           
            #     'max_depth':[3,4,5],                             

            # },
            # "AdaBoost": {
            #    'learning_rate': [0.1, 0.05, 0.01],
            #    'n_estimators': [50, 100, 200,300],                    # Low estimators make AdaBoost underperform
            # },
    #         "Logistic Regression": {
    #         'penalty': ['l2'],                                 # Regularization type
    #         'C': [0.01, 0.1, 1.0, 10.0],                       # Inverse regularization strength (smaller = less overfitting)
    #         'solver': ['lbfgs', 'saga'],                       # Optimization algorithms
    #         'max_iter': [100, 500]                             # Gives the solver enough time to converge
    # } 

        "CatBoost": {
        "iterations": [400],
        "learning_rate": [0.05],
        "depth": [6],
        "l2_leaf_reg": [3],
        "bootstrap_type": ["Bernoulli"],
        "subsample": [0.8],
        "random_state": [42],
        "verbose": [0]
    }
}
    # "Hist Gradient Boosting": {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'max_iter': [100, 200],
    #     'max_depth': [3, 5, 7]
    # },
#     "XGBoost": {
#     'learning_rate': [0.03, 0.05, 0.1],          # Shifting away from 0.01 (often too slow for tabular depth)
#     'n_estimators': [100, 200, 300],             # Added 300 to match lower learning rates
#     'max_depth': [3, 5, 6],                      # 7 can overfit tabular data easily; 6 is often the sweet spot
#     'subsample': [0.8, 0.9],                     # Added 0.9 for slight variance changes
#     'colsample_bytree': [0.8, 1.0],              # CRUCIAL: Fraction of features evaluated per tree (prevents dominant features from taking over)
#     'gamma': [0, 0.1, 0.2]                       # Minimum loss reduction required to make a split (adds structural pruning)
# }
    # "LightGBM": {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5, 7]
    # }

            
        
        try:
            model_report=evaluate_models(x_train,x_test,y_train,y_test,models,params,file_path)

            best_model_name=max(model_report ,key=lambda x:model_report[x]["test_score"])
            best_model_score=model_report[best_model_name]['train_score']
            best_model=model_report[best_model_name]["best_model"]
 
            y_train_pred=best_model.predict(x_train)
            classification_train_metric=calculate_metrics(y_train,y_train_pred)
            # self.track_mlflow(best_model,classification_train_metric,"train")

            y_test_pred=best_model.predict(x_test)
            classification_test_metric=calculate_metrics(y_test,y_test_pred)
            # self.track_mlflow(best_model,classification_test_metric,"test")

            save_object(self.model_trainer_config.trained_model_file_path,best_model)
       
            model_trainer_artifact=ModelTrainerArtifact(
              model_obj_file_path=self.model_trainer_config.trained_model_file_path,
              train_metric_artifact = classification_train_metric,
              test_metric_artifact = classification_test_metric
            )
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
     
    def initiate_model_training(self):
        try:
            train_transformed_data=load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_transformed_data=load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)

            x_train,y_train,x_test,y_test=(
             train_transformed_data[:,:-1],
             train_transformed_data[:,-1],
             test_transformed_data[:,:-1],
             test_transformed_data[:,-1]
        )
            model_train_artifact=self.train_model(x_train,x_test,y_train,y_test,self.model_trainer_config.model_evaluation_dir)
            return model_train_artifact
        except Exception as e:
            raise CustomException(e,sys)

    

        
        
             
             

          

          