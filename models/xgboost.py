# models/xgboost_model.py

from xgboost import XGBClassifier
from models.traditional_models_base import TraditionalCKDClassifier

class XGBoostCKDClassifier(TraditionalCKDClassifier):
    """XGBoost Classifier for CKD."""
    
    def __init__(self, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1):
        super().__init__(random_state)
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='mlogloss',  # For multi-class classification
            use_label_encoder=False,  # Suppress warning
            n_jobs=-1  # Use all available cores
        )