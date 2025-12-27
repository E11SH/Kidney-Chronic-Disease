"""
Model Loader - Handles loading and managing trained ML models
"""

import os
import sys
import pandas as pd

# Add parent directory to import model classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.logistic_regression import LogisticRegressionCKDClassifier
from models.random_forest import RandomForestCKDClassifier
from models.xgboost import XGBoostCKDClassifier
from models.nn import NeuralNetworkCKDClassifier
from models.knn import KNNCKDClassifier
from models.svm import SupportVectorMachineCKDClassifier


class ModelLoader:
    """Manages loading and caching of trained models"""
    
    def __init__(self, models_dir):
        """
        Initialize Model Loader
        
        Parameters:
        -----------
        models_dir : str
            Path to directory containing .pkl model files
        """
        self.models_dir = models_dir
        self.loaded_models = {}
        
        # Model class mapping
        self.model_classes = {
            'logistic_regression': LogisticRegressionCKDClassifier,
            'random_forest': RandomForestCKDClassifier,
            'xgboost': XGBoostCKDClassifier,
            'neural_network': NeuralNetworkCKDClassifier,
            'knn': KNNCKDClassifier,
            'svm': SupportVectorMachineCKDClassifier
        }
        
        # Model metadata (from your CV results)
        self.model_metadata = {
            'random_forest': {
                'name': 'Random Forest',
                'accuracy': 0.9925,
                'f1_score': 0.9920,
                'roc_auc': 1.0000,
                'description': 'Ensemble learning method, best overall performance'
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'accuracy': 0.9900,
                'f1_score': 0.9895,
                'roc_auc': 1.0000,
                'description': 'Linear classifier, highly interpretable'
            },
            'svm': {
                'name': 'Support Vector Machine',
                'accuracy': 0.9875,
                'f1_score': 0.9869,
                'roc_auc': 1.0000,
                'description': 'Kernel-based classifier, robust decision boundaries'
            },
            'xgboost': {
                'name': 'XGBoost',
                'accuracy': 0.9875,
                'f1_score': 0.9866,
                'roc_auc': 0.9984,
                'description': 'Gradient boosting, excellent performance'
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'accuracy': 0.9575,
                'f1_score': 0.9560,
                'roc_auc': 0.9940,
                'description': 'Instance-based learning, simple yet effective'
            },
            'neural_network': {
                'name': 'Neural Network',
                'accuracy': 0.9450,
                'f1_score': 0.9431,
                'roc_auc': 0.9997,
                'description': 'Deep learning model with 3 hidden layers'
            }
        }
    
    def load_model(self, model_name):
        """
        Load a single model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load (e.g., 'random_forest')
        
        Returns:
        --------
        model : TraditionalCKDClassifier
            Loaded model instance
        """
        # Check if already loaded (caching)
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Check if model class exists
        if model_name not in self.model_classes:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create model instance
        model_class = self.model_classes[model_name]
        model_instance = model_class()
        
        # Load from file
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        success = model_instance.load_model(model_path)
        
        if not success:
            raise Exception(f"Failed to load model: {model_name}")
        
        # Cache the loaded model
        self.loaded_models[model_name] = model_instance
        
        return model_instance
    
    def load_all_models(self):
        """
        Load all available models
        
        Returns:
        --------
        loaded : list
            List of successfully loaded model names
        """
        loaded = []
        
        for model_name in self.model_classes.keys():
            try:
                self.load_model(model_name)
                loaded.append(model_name)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {model_name}: {str(e)}")
        
        return loaded
    
    def get_model(self, model_name):
        """
        Get a loaded model (loads if not cached)
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        
        Returns:
        --------
        model : TraditionalCKDClassifier
            Model instance
        """
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        return self.loaded_models[model_name]
    
    def get_models_info(self):
        """
        Get information about all available models
        
        Returns:
        --------
        info : list
            List of dictionaries with model information
        """
        models_info = []
        
        for model_name, metadata in self.model_metadata.items():
            # Check if model file exists
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            available = os.path.exists(model_path)
            
            models_info.append({
                'id': model_name,
                'name': metadata['name'],
                'accuracy': metadata['accuracy'],
                'f1_score': metadata['f1_score'],
                'roc_auc': metadata['roc_auc'],
                'description': metadata['description'],
                'available': available,
                'loaded': model_name in self.loaded_models
            })
        
        # Sort by accuracy (best first)
        models_info.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return models_info
    
    def get_model_details(self, model_name):
        """
        Get detailed information about a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        
        Returns:
        --------
        details : dict
            Detailed model information
        """
        if model_name not in self.model_metadata:
            return None
        
        metadata = self.model_metadata[model_name]
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        details = {
            'id': model_name,
            'name': metadata['name'],
            'accuracy': metadata['accuracy'],
            'f1_score': metadata['f1_score'],
            'roc_auc': metadata['roc_auc'],
            'description': metadata['description'],
            'file_path': model_path,
            'available': os.path.exists(model_path),
            'loaded': model_name in self.loaded_models,
            'file_size': os.path.getsize(model_path) if os.path.exists(model_path) else None
        }
        
        return details
    
    def unload_model(self, model_name):
        """
        Unload a model from cache
        
        Parameters:
        -----------
        model_name : str
            Name of the model to unload
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
    
    def unload_all_models(self):
        """Unload all models from cache"""
        self.loaded_models.clear()
    
    def get_loaded_count(self):
        """Get number of loaded models"""
        return len(self.loaded_models)