# models/neural_network_model.py

from sklearn.neural_network import MLPClassifier
from models.traditional_models_base import TraditionalCKDClassifier

class NeuralNetworkCKDClassifier(TraditionalCKDClassifier):
    """Multi-Layer Perceptron (Neural Network) Classifier for CKD."""
    
    def __init__(self, random_state=42, hidden_layer_sizes=(128, 64, 32), 
                 max_iter=300, learning_rate_init=0.001, alpha=0.0001):
        super().__init__(random_state)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,  # 3 hidden layers: 128 -> 64 -> 32 neurons
            activation='relu',                       # ReLU activation function
            solver='adam',                          # Adam optimizer
            alpha=alpha,                            # L2 regularization parameter
            batch_size='auto',                      # Batch size (auto = min(200, n_samples))
            learning_rate='adaptive',               # Adaptive learning rate
            learning_rate_init=learning_rate_init,  # Initial learning rate
            max_iter=max_iter,                      # Maximum epochs
            random_state=random_state,
            early_stopping=True,                    # Enable early stopping
            validation_fraction=0.1,                # 10% for validation during training
            n_iter_no_change=10,                    # Stop if no improvement for 10 epochs
            verbose=False                           # Set to True to see training progress
        )