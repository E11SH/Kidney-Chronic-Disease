from sklearn.ensemble import RandomForestClassifier
from models.traditional_models_base import TraditionalCKDClassifier

class RandomForestCKDClassifier(TraditionalCKDClassifier):
    """Random Forest Classifier for CKD."""
    
    def __init__(self, random_state=42, n_estimators=100, max_depth=10):
        super().__init__(random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state, 
            n_jobs=-1 # Use all available cores
        )