import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Safely drop 'id' or MongoDB's '_id' if present
        cols_to_drop = [col for col in ['id', '_id', 'ID'] if col in X.columns]
        if cols_to_drop:
            X.drop(columns=cols_to_drop, inplace=True)

        # 2. Age Conversion (Days to Years)
        if 'age' in X.columns:
            X['age'] = (X['age'] / 365.0).round(2)

        # 3. Outlier Removal based on Quantiles & Validation
        low_ap_hi, high_ap_hi = X['ap_hi'].quantile(0.001), X['ap_hi'].quantile(0.999)
        low_ap_lo, high_ap_lo = X['ap_lo'].quantile(0.001), X['ap_lo'].quantile(0.999)
        low_height, high_height = X['height'].quantile(0.001), X['height'].quantile(0.999)
        low_weight, high_weight = X['weight'].quantile(0.001), X['weight'].quantile(0.999)

        X = X[
            (X['ap_hi'] >= low_ap_hi) & (X['ap_hi'] <= high_ap_hi) &
            (X['ap_lo'] >= low_ap_lo) & (X['ap_lo'] <= high_ap_lo) &
            (X['height'] >= low_height) & (X['height'] <= high_height) &
            (X['weight'] >= low_weight) & (X['weight'] <= high_weight) &
            (X['ap_hi'] > X['ap_lo'])  # Ensure Systolic > Diastolic
        ]

        X = X.reset_index(drop=True)

        return X