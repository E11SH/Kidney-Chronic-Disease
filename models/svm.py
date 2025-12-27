from sklearn.svm import SVC
from models.traditional_models_base import TraditionalCKDClassifier

class SupportVectorMachineCKDClassifier(TraditionalCKDClassifier):
    """
    Support Vector Machine Classifier for CKD prediction.
    """
    
    def __init__(self, random_state=42, kernel='linear', C=1.0, probability=True):
        """
        Initialize SVM Classifier
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        kernel : str
            Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C : float
            Regularization parameter
        probability : bool
            Whether to enable probability estimates
        """
        super().__init__(random_state)
        self.kernel = kernel
        self.C = C
        self.probability = probability
        
        # Initialize the model
        self.model = SVC(
            kernel=self.kernel, 
            C=self.C,
            probability=self.probability,
            random_state=self.random_state
        )