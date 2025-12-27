from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add parent directory to path to import model classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'trained_models')

# Feature names expected by the models (from UCI CKD dataset)
FEATURE_NAMES = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 
    'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# Default values for missing features (medians/modes from training data)
FEATURE_DEFAULTS = {
    'age': 51.0,
    'bp': 76.0,
    'sg': 1.017,
    'al': 1.0,
    'su': 0.0,
    'rbc': 'normal',
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': 121.0,
    'bu': 40.0,
    'sc': 1.2,
    'sod': 137.0,
    'pot': 4.5,
    'hemo': 12.5,
    'pcv': 38.0,
    'wbcc': 8400.0,
    'rbcc': 4.7,
    'htn': 'no',
    'dm': 'no',
    'cad': 'no',
    'appet': 'good',
    'pe': 'no',
    'ane': 'no'
}

# Load models
MODELS = {}

def load_models():
    """Load all trained models"""
    global MODELS
    model_files = {
        'logistic_regression': 'logistic_regression.pkl',
        'random_forest': 'random_forest.pkl',
        'xgboost': 'xgboost.pkl',
        'neural_network': 'neural_network.pkl',
        'knn': 'knn.pkl',
        'svm': 'svm.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                MODELS[name] = joblib.load(path)
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ Model file not found: {path}")

def prepare_features(input_data):
    """
    Prepare features for prediction by filling missing values with defaults
    
    Args:
        input_data: dict with user-provided features
    
    Returns:
        DataFrame with all 24 features
    """
    # Start with defaults
    features = FEATURE_DEFAULTS.copy()
    
    # Update with user-provided values
    for key, value in input_data.items():
        if key in features:
            # Handle empty strings and None
            if value == '' or value is None:
                continue
            features[key] = value
    
    # Create DataFrame with single row
    df = pd.DataFrame([features])
    
    # Ensure correct order
    df = df[FEATURE_NAMES]
    
    return df

def preprocess_for_model(df):
    """
    Apply same preprocessing as training:
    - Encode categorical variables
    - Handle data types
    """
    df = df.copy()
    
    # Categorical columns that need encoding
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    
    # Encode categorical variables (simple label encoding)
    encoding_map = {
        'rbc': {'normal': 0, 'abnormal': 1},
        'pc': {'normal': 0, 'abnormal': 1},
        'pcc': {'notpresent': 0, 'present': 1},
        'ba': {'notpresent': 0, 'present': 1},
        'htn': {'no': 0, 'yes': 1},
        'dm': {'no': 0, 'yes': 1},
        'cad': {'no': 0, 'yes': 1},
        'appet': {'good': 0, 'poor': 1},
        'pe': {'no': 0, 'yes': 1},
        'ane': {'no': 0, 'yes': 1}
    }
    
    for col, mapping in encoding_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Ensure all numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
    return df

def make_prediction(model, features_df):
    """Make prediction and return detailed results"""
    try:
        # Get prediction
        prediction = model.predict(features_df)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            probabilities = {
                'No Disease': float(proba[0]),
                'CKD': float(proba[1])
            }
            confidence_score = float(max(proba))
        else:
            probabilities = {
                'No Disease': 1.0 if prediction == 0 else 0.0,
                'CKD': 1.0 if prediction == 1 else 0.0
            }
            confidence_score = 1.0
        
        # Determine confidence level
        if confidence_score >= 0.9:
            confidence = 'high'
        elif confidence_score >= 0.7:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        result = {
            'prediction': 'CKD' if prediction == 1 else 'No Disease',
            'prediction_code': int(prediction),
            'probabilities': probabilities,
            'confidence': confidence,
            'confidence_score': confidence_score
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'CKD Prediction API',
        'version': '1.0.0',
        'models_loaded': list(MODELS.keys()),
        'endpoints': {
            '/api/models': 'GET - List available models',
            '/api/predict': 'POST - Single prediction',
            '/api/predict/batch': 'POST - Batch prediction (CSV)',
            '/api/features': 'GET - Feature information'
        }
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_info = []
    
    # Approximate accuracies from your results
    accuracies = {
        'random_forest': 99.25,
        'logistic_regression': 99.00,
        'svm': 98.75,
        'xgboost': 98.75,
        'knn': 95.75,
        'neural_network': 94.50
    }
    
    for name in MODELS.keys():
        models_info.append({
            'name': name,
            'display_name': name.replace('_', ' ').title(),
            'accuracy': accuracies.get(name, 95.0),
            'loaded': True
        })
    
    return jsonify({
        'success': True,
        'models': models_info,
        'total': len(models_info)
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature information and expected format"""
    feature_info = {
        'age': {'type': 'numeric', 'unit': 'years', 'range': '18-100', 'default': 51.0},
        'bp': {'type': 'numeric', 'unit': 'mm/Hg', 'range': '50-180', 'default': 76.0},
        'sg': {'type': 'numeric', 'unit': 'specific gravity', 'range': '1.005-1.025', 'default': 1.017},
        'al': {'type': 'numeric', 'unit': 'albumin', 'range': '0-5', 'default': 1.0},
        'su': {'type': 'numeric', 'unit': 'sugar', 'range': '0-5', 'default': 0.0},
        'rbc': {'type': 'categorical', 'values': ['normal', 'abnormal'], 'default': 'normal'},
        'pc': {'type': 'categorical', 'values': ['normal', 'abnormal'], 'default': 'normal'},
        'pcc': {'type': 'categorical', 'values': ['present', 'notpresent'], 'default': 'notpresent'},
        'ba': {'type': 'categorical', 'values': ['present', 'notpresent'], 'default': 'notpresent'},
        'bgr': {'type': 'numeric', 'unit': 'mg/dL', 'range': '20-490', 'default': 121.0},
        'bu': {'type': 'numeric', 'unit': 'mg/dL', 'range': '1.5-391', 'default': 40.0},
        'sc': {'type': 'numeric', 'unit': 'mg/dL', 'range': '0.4-76', 'default': 1.2},
        'sod': {'type': 'numeric', 'unit': 'mEq/L', 'range': '4.5-163', 'default': 137.0},
        'pot': {'type': 'numeric', 'unit': 'mEq/L', 'range': '2.5-47', 'default': 4.5},
        'hemo': {'type': 'numeric', 'unit': 'g/dL', 'range': '3.1-17.8', 'default': 12.5},
        'pcv': {'type': 'numeric', 'unit': '%', 'range': '9-54', 'default': 38.0},
        'wbcc': {'type': 'numeric', 'unit': 'cells/cumm', 'range': '2200-26400', 'default': 8400.0},
        'rbcc': {'type': 'numeric', 'unit': 'millions/cmm', 'range': '2.1-8', 'default': 4.7},
        'htn': {'type': 'categorical', 'values': ['yes', 'no'], 'default': 'no'},
        'dm': {'type': 'categorical', 'values': ['yes', 'no'], 'default': 'no'},
        'cad': {'type': 'categorical', 'values': ['yes', 'no'], 'default': 'no'},
        'appet': {'type': 'categorical', 'values': ['good', 'poor'], 'default': 'good'},
        'pe': {'type': 'categorical', 'values': ['yes', 'no'], 'default': 'no'},
        'ane': {'type': 'categorical', 'values': ['yes', 'no'], 'default': 'no'}
    }
    
    return jsonify({
        'success': True,
        'features': feature_info,
        'total_features': len(feature_info)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a single prediction
    
    Expected JSON format:
    {
        "model": "random_forest",
        "features": {
            "age": 65,
            "bp": 80,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_name = data.get('model', 'random_forest')
        features = data.get('features', {})
        
        if not features:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            }), 400
        
        if model_name not in MODELS:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not found'
            }), 404
        
        # Prepare features (fill missing with defaults)
        features_df = prepare_features(features)
        
        # Preprocess
        features_processed = preprocess_for_model(features_df)
        
        # Make prediction
        result = make_prediction(MODELS[model_name], features_processed)
        result['model_used'] = model_name
        result['features_received'] = len([v for v in features.values() if v not in ['', None]])
        result['total_features'] = 24
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions from CSV file
    
    Expected form-data:
    - file: CSV file with 24 features
    - model: model name (optional, default: random_forest)
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        model_name = request.form.get('model', 'random_forest')
        
        if model_name not in MODELS:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not found'
            }), 404
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read CSV: {str(e)}'
            }), 400
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'CSV file is empty'
            }), 400
        
        # Process each row
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict
                row_dict = row.to_dict()
                
                # Prepare features
                features_df = prepare_features(row_dict)
                
                # Preprocess
                features_processed = preprocess_for_model(features_df)
                
                # Make prediction
                prediction_result = make_prediction(MODELS[model_name], features_processed)
                
                # Add row info
                result_row = {
                    'row_number': int(idx + 1),
                    'prediction': prediction_result['prediction'],
                    'prediction_code': prediction_result['prediction_code'],
                    'probabilities': prediction_result['probabilities'],
                    'confidence': prediction_result['confidence'],
                    'confidence_score': prediction_result['confidence_score'],
                    'input_features': {k: v for k, v in row_dict.items() if pd.notna(v)}
                }
                
                results.append(result_row)
                
            except Exception as e:
                results.append({
                    'row_number': int(idx + 1),
                    'error': str(e),
                    'prediction': 'Error'
                })
        
        # Calculate summary statistics
        successful = [r for r in results if 'error' not in r]
        ckd_count = sum(1 for r in successful if r['prediction'] == 'CKD')
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_rows': len(results),
                'successful_predictions': len(successful),
                'failed_predictions': len(results) - len(successful),
                'ckd_detected': ckd_count,
                'no_disease': len(successful) - ckd_count
            },
            'model_used': model_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/feature/importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance for tree-based models"""
    try:
        model_name = request.args.get('model', 'random_forest')
        
        if model_name not in MODELS:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not found'
            }), 404
        
        model = MODELS[model_name]
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature importance dict
            feature_importance = {}
            for name, importance in zip(FEATURE_NAMES, importances):
                feature_importance[name] = float(importance)
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return jsonify({
                'success': True,
                'feature_importance': feature_importance,
                'model': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} does not support feature importance'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("="*80)
    print("🚀 CKD PREDICTION API SERVER")
    print("="*80)
    print(f"\n📁 Models directory: {MODELS_DIR}")
    
    # Load all models on startup
    load_models()
    
    print(f"\n✓ Successfully loaded {len(MODELS)} models:")
    for model_name in MODELS.keys():
        print(f"  - {model_name}")
    
    print(f"\n🌐 Server starting on http://localhost:5000")
    print("="*80)
    print("\n📚 Available endpoints:")
    print("  GET  /api/models          - List all models")
    print("  GET  /api/features        - Get feature information")
    print("  POST /api/predict         - Single prediction")
    print("  POST /api/predict/batch   - Batch prediction (CSV)")
    print("  GET  /api/feature/importance?model=<name> - Feature importance")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)