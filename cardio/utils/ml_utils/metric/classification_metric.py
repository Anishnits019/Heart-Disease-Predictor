import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from cardio.entity.artifact_entity import ClassificationMetricArtifact
from cardio.exception.exception import CustomException

def calculate_metrics(y_true, y_pred, y_prob=None) -> ClassificationMetricArtifact:
    """
    Calculates classification evaluation metrics and returns a ClassificationMetricArtifact.
    
    :param y_true: Ground truth target values
    :param y_pred: Predicted class labels
    :param y_prob: (Optional) Predicted probability estimates for ROC-AUC calculation
    """
    try:
        # Calculate evaluation scores
        model_f1_score = f1_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_accuracy_score = accuracy_score(y_true, y_pred)
        
        # Calculate ROC-AUC score if probabilities are provided
        model_roc_auc_score = None
        if y_prob is not None:
            try:
                model_roc_auc_score = roc_auc_score(y_true, y_prob)
            except Exception:
                model_roc_auc_score = None

        # Instantiate artifact (with correct variable mapping)
        classification_artifact = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,  # Fixed mapping
            recall_score=model_recall_score,        # Fixed mapping
            accuracy_score=model_accuracy_score,
            roc_auc_score=model_roc_auc_score
        )
        return classification_artifact

    except Exception as e:
        raise CustomException(e, sys)