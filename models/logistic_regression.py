from sklearn.linear_model import LogisticRegression
from models.traditional_models_base import TraditionalCKDClassifier

class LogisticRegressionCKDClassifier(TraditionalCKDClassifier):
    """Logistic Regression Classifier for CKD."""
    
    def __init__(self, random_state=42, C=1.0, solver='lbfgs'):
        super().__init__(random_state)
        # Use 'ovr' (One-vs-Rest) strategy which liblinear supports for multi-class
        self.model = LogisticRegression(
            C=C, 
            solver=solver, 
            random_state=random_state, 
            multi_class='multinomial',
            max_iter=1000
        )