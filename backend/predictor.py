"""
CKD Predictor - Handles prediction logic and feature processing
"""

import pandas as pd
import numpy as np


class CKDPredictor:
    """Handles predictions using loaded models"""
    
    def __init__(self, model_loader):
        """
        Initialize predictor
        
        Parameters:
        -----------
        model_loader : ModelLoader
            Instance of ModelLoader class
        """
        self.model_loader = model_loader
        
        # Feature names in correct order
        self.feature_names = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
            'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
    
    def prepare_features(self, features_dict):
        """
        Prepare features for prediction
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary of feature names and values
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with features in correct order
        """
        # Create DataFrame with all features
        df = pd.DataFrame([features_dict])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                # Set default values for missing features
                df[feature] = self._get_default_value(feature)
        
        # Select only required features in correct order
        df = df[self.feature_names]
        
        return df
    
    def _get_default_value(self, feature_name):
        """
        Get default value for missing feature
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature
        
        Returns:
        --------
        default : float or str
            Default value
        """
        # Categorical features - default to most common value
        categorical_defaults = {
            'rbc': 'normal',
            'pc': 'normal',
            'pcc': 'notpresent',
            'ba': 'notpresent',
            'htn': 'no',
            'dm': 'no',
            'cad': 'no',
            'appet': 'good',
            'pe': 'no',
            'ane': 'no'
        }
        
        if feature_name in categorical_defaults:
            return categorical_defaults[feature_name]
        
        # Numerical features - default to median from training data
        numerical_defaults = {
            'age': 50,
            'bp': 80,
            'sg': 1.020,
            'al': 0,
            'su': 0,
            'bgr': 120,
            'bu': 40,
            'sc': 1.0,
            'sod': 140,
            'pot': 4.5,
            'hemo': 14,
            'pcv': 42,
            'wbcc': 8000,
            'rbcc': 4.5
        }
        
        return numerical_defaults.get(feature_name, 0)
    
    def predict_single(self, model_name, features_dict):
        """
        Make a single prediction
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        features_dict : dict
            Dictionary of features
        
        Returns:
        --------
        result : dict
            Prediction result with probabilities
        """
        # Get model
        model = self.model_loader.get_model(model_name)
        
        # Prepare features
        input_df = self.prepare_features(features_dict)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Get probabilities
        probabilities, class_names = model.predict_proba(input_df)
        
        # Format result
        result = {
            'prediction': prediction,
            'probabilities': {
                class_names[0]: float(probabilities[0][0]),
                class_names[1]: float(probabilities[0][1])
            },
            'model_used': model_name,
            'confidence': self._calculate_confidence(probabilities[0])
        }
        
        return result
    
    def predict_batch(self, model_name, dataframe):
        """
        Make batch predictions
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        dataframe : pd.DataFrame
            DataFrame with features
        
        Returns:
        --------
        results : list
            List of prediction results
        """
        # Get model
        model = self.model_loader.get_model(model_name)
        
        results = []
        
        for idx, row in dataframe.iterrows():
            features_dict = row.to_dict()
            
            try:
                result = self.predict_single(model_name, features_dict)
                result['row_index'] = int(idx)
                results.append(result)
            except Exception as e:
                results.append({
                    'row_index': int(idx),
                    'error': str(e)
                })
        
        return results
    
    def _calculate_confidence(self, probabilities):
        """
        Calculate confidence level from probabilities
        
        Parameters:
        -----------
        probabilities : np.array
            Array of class probabilities
        
        Returns:
        --------
        confidence : str
            'high', 'medium', or 'low'
        """
        max_prob = max(probabilities)
        
        if max_prob >= 0.9:
            return 'high'
        elif max_prob >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for tree-based models
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        
        Returns:
        --------
        importance : dict or None
            Dictionary of feature names and importance scores
        """
        if model_name not in ['random_forest', 'xgboost']:
            return None
        
        try:
            model = self.model_loader.get_model(model_name)
            
            # Get feature importances from the trained classifier
            if hasattr(model.trained_classifier, 'feature_importances_'):
                importances = model.trained_classifier.feature_importances_
                
                # Map to feature names
                # Note: After preprocessing, feature names might be different
                # For now, use the original feature names
                importance_dict = {}
                for i, feature in enumerate(self.feature_names):
                    if i < len(importances):
                        importance_dict[feature] = float(importances[i])
                
                # Sort by importance
                importance_dict = dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
                
                return importance_dict
            else:
                return None
                
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
    
    def explain_prediction(self, model_name, features_dict, prediction_result):
        """
        Generate explanation for a prediction
        
        Parameters:
        -----------
        model_name : str
            Name of the model used
        features_dict : dict
            Input features
        prediction_result : dict
            Prediction result
        
        Returns:
        --------
        explanation : dict
            Detailed explanation of the prediction
        """
        # Get feature importance
        importance = self.get_feature_importance(model_name)
        
        # Identify key features that influenced the prediction
        key_features = []
        
        if importance:
            # Get top 5 most important features
            top_features = list(importance.items())[:5]
            
            for feature_name, importance_score in top_features:
                if feature_name in features_dict:
                    key_features.append({
                        'feature': feature_name,
                        'value': features_dict[feature_name],
                        'importance': importance_score
                    })
        
        explanation = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'key_features': key_features,
            'interpretation': self._generate_interpretation(
                prediction_result, key_features
            )
        }
        
        return explanation
    
    def _generate_interpretation(self, prediction_result, key_features):
        """
        Generate human-readable interpretation
        
        Parameters:
        -----------
        prediction_result : dict
            Prediction result
        key_features : list
            List of key features
        
        Returns:
        --------
        interpretation : str
            Human-readable explanation
        """
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        if prediction == 'CKD':
            base_text = f"The model predicts CKD with {confidence} confidence. "
        else:
            base_text = f"The model predicts No CKD with {confidence} confidence. "
        
        if key_features:
            top_feature = key_features[0]
            feature_text = f"The most influential factor was {top_feature['feature']} "
            feature_text += f"with a value of {top_feature['value']}. "
            return base_text + feature_text
        
        return base_text