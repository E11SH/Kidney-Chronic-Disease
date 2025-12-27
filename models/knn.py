from sklearn.neighbors import KNeighborsClassifier
from models.traditional_models_base import TraditionalCKDClassifier

class KNNCKDClassifier(TraditionalCKDClassifier):
    """
    K-Nearest Neighbors Classifier for CKD prediction.
    """
    
    def __init__(self, random_state=42, n_neighbors=5, weights='uniform'):
        """
        Initialize KNN Classifier
        
        Parameters:
        -----------
        random_state : int
            Random seed (unused by KNN itself but kept for interface consistency)
        n_neighbors : int
            Number of neighbors to use
        weights : str
            Weight function used in prediction ('uniform', 'distance')
        """
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
        self.weights = weights
        
        # Initialize the model
        # KNN doesn't use random_state in init, but we pass it if needed by other components, 
        # though standard KNeighborsClassifier doesn't take it.
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )