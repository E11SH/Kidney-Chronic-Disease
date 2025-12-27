"""
UCI Chronic Kidney Disease Dataset - Cross-Validation Training Pipeline
========================================================================
Integrates with existing project structure for proper CV evaluation
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import CLEANED_DATA_PATH, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

# Import data cleaning
from data_cleaning import full_cleaning_pipeline

# Import all model classes
from models.logistic_regression import LogisticRegressionCKDClassifier
from models.random_forest import RandomForestCKDClassifier
from models.xgboost import XGBoostCKDClassifier
from models.nn import NeuralNetworkCKDClassifier
from models.knn import KNNCKDClassifier
from models.svm import SupportVectorMachineCKDClassifier


def load_and_prepare_data(use_cleaned=True):
    """
    Load and prepare data for training
    
    Parameters:
    -----------
    use_cleaned : bool
        If True, load cleaned data. If False, load raw and clean it.
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (encoded)
    label_encoder : LabelEncoder
        Fitted label encoder for target
    """
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    if use_cleaned and os.path.exists(CLEANED_DATA_PATH):
        print(f"\n✓ Loading cleaned data from: {CLEANED_DATA_PATH}")
        df = pd.read_csv(CLEANED_DATA_PATH)
    else:
        print(f"\n✓ Loading raw data and applying cleaning pipeline...")
        from data_loader import load_data
        df_raw = load_data()
        df, _ = full_cleaning_pipeline(df_raw, target_col='class')
    
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Clean and encode target
    print("\n🎯 Preparing target variable...")
    df['class'] = df['class'].str.strip().str.lower()
    df = df[df['class'].isin(['ckd', 'notckd'])]
    print(f"✓ Target cleaned. Remaining rows: {len(df)}")
    
    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Class distribution:")
    for cls, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        count = sum(y_encoded == label)
        print(f"   {cls}: {count} ({count/len(y_encoded)*100:.1f}%)")
    
    print(f"✓ Features: {X.shape[1]} columns")
    print(f"✓ Target encoded: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return X, y_encoded, label_encoder


def prepare_features_for_cv(X):
    """
    Prepare features for cross-validation
    Handles encoding that will be done inside each CV fold
    
    Parameters:
    -----------
    X : pd.DataFrame
        Raw feature matrix
    
    Returns:
    --------
    X_prepared : pd.DataFrame
        Feature matrix ready for CV (with categorical encoding)
    """
    print("\n🔧 Preparing features for cross-validation...")
    
    X_prepared = X.copy()
    
    # Handle missing values
    print("   Handling missing values...")
    for col in X_prepared.select_dtypes(include=[np.number]).columns:
        if X_prepared[col].isnull().any():
            X_prepared[col].fillna(X_prepared[col].median(), inplace=True)
    
    for col in X_prepared.select_dtypes(include=['object']).columns:
        if X_prepared[col].isnull().any():
            mode_val = X_prepared[col].mode()[0] if len(X_prepared[col].mode()) > 0 else 'unknown'
            X_prepared[col].fillna(mode_val, inplace=True)
    
    # Encode categorical features
    print("   Encoding categorical features...")
    categorical_cols = X_prepared.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X_prepared[col] = le.fit_transform(X_prepared[col].astype(str))
        print(f"   ✓ Encoded {len(categorical_cols)} categorical columns")
    
    print(f"✓ Features prepared: {X_prepared.shape}")
    return X_prepared


def create_model_instances():
    """
    Create instances of all models to train
    
    Returns:
    --------
    models : dict
        Dictionary of model name -> model instance
    """
    return {
        'Logistic Regression': LogisticRegressionCKDClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestCKDClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=5),
        'XGBoost': XGBoostCKDClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=3),
        'Neural Network': NeuralNetworkCKDClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=(128, 64, 32)),
        'KNN': KNNCKDClassifier(random_state=RANDOM_STATE, n_neighbors=5),
        'SVM': SupportVectorMachineCKDClassifier(random_state=RANDOM_STATE, kernel='rbf')
    }


def perform_cross_validation(X, y):
    """
    Perform cross-validation on all models
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : np.array
        Target variable (encoded)
    
    Returns:
    --------
    results : dict
        Dictionary with model results
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION TRAINING")
    print("="*80)
    
    models = create_model_instances()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1_macro',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'roc_auc': 'roc_auc'
    }
    
    print(f"\n{'Model':<20} {'Test Acc':<15} {'Test F1':<15} {'ROC-AUC':<12} {'Status'}")
    print("-" * 75)
    
    results = {}
    
    for model_name, model_instance in models.items():
        try:
            # Create a simple wrapper that uses the model's sklearn classifier
            # We need to extract just the sklearn model for cross_validate
            
            # Temporarily fit preprocessor to get the pipeline ready
            # For CV, we'll create a custom scoring approach
            from sklearn.pipeline import Pipeline
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE
            from sklearn.preprocessing import StandardScaler
            
            # Create pipeline with SMOTE
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', model_instance.model)
            ])
            
            cv_results = cross_validate(
                pipeline, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            test_acc = cv_results['test_accuracy'].mean()
            test_acc_std = cv_results['test_accuracy'].std()
            train_acc = cv_results['train_accuracy'].mean()
            test_f1 = cv_results['test_f1'].mean()
            test_f1_std = cv_results['test_f1'].std()
            test_auc = cv_results['test_roc_auc'].mean()
            
            overfit_gap = train_acc - test_acc
            status = "🚨 OVERFIT" if overfit_gap > 0.10 else "✓ OK"
            
            results[model_name] = {
                'test_acc': test_acc,
                'test_acc_std': test_acc_std,
                'train_acc': train_acc,
                'test_f1': test_f1,
                'test_f1_std': test_f1_std,
                'test_auc': test_auc,
                'test_precision': cv_results['test_precision'].mean(),
                'test_recall': cv_results['test_recall'].mean(),
                'overfit_gap': overfit_gap
            }
            
            print(f"{model_name:<20} {test_acc:.4f}±{test_acc_std:.3f}  "
                  f"{test_f1:.4f}±{test_f1_std:.3f}  {test_auc:.4f}    {status}")
        
        except Exception as e:
            print(f"{model_name:<20} Error: {str(e)[:40]}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['test_f1'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['test_acc']:.4f} ± {best_model[1]['test_acc_std']:.4f}")
        print(f"   F1-Score: {best_model[1]['test_f1']:.4f} ± {best_model[1]['test_f1_std']:.4f}")
        print(f"   ROC-AUC:  {best_model[1]['test_auc']:.4f}")
    
    return results


def train_final_models(X, y):
    """
    Train final models on full dataset for deployment
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : np.array
        Target variable (encoded)
    
    Returns:
    --------
    trained_models : dict
        Dictionary of trained model instances
    """
    print("\n" + "="*80)
    print("TRAINING FINAL MODELS (Full Dataset)")
    print("="*80)
    print("\nTraining models on full dataset for deployment...")
    
    models = create_model_instances()
    trained_models = {}
    
    # Create a temporary DataFrame with target for the train method
    df_full = X.copy()
    df_full['class'] = y
    
    for model_name, model_instance in models.items():
        print(f"\n   Training {model_name}...")
        try:
            # Train on full dataset with SMOTE
            model_instance.train(
                df_full, 
                target_col='class',
                test_size=0.2,  # Still need validation for training
                verbose=False,
                use_smote=True,
                noise_level=0.0
            )
            trained_models[model_name] = model_instance
            print(f"   ✓ {model_name} trained successfully")
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {str(e)}")
    
    return trained_models


def save_models(trained_models):
    """
    Save all trained models to disk
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained model instances
    """
    print("\n" + "="*80)
    print("SAVING TRAINED MODELS")
    print("="*80)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_name, model_instance in trained_models.items():
        safe_name = model_name.lower().replace(' ', '_')
        filepath = f"{MODELS_DIR}{safe_name}.pkl"
        model_instance.save_model(filepath)
    
    print(f"\n✓ All models saved to: {MODELS_DIR}")


def save_cv_results(results):
    """
    Save cross-validation results to CSV
    
    Parameters:
    -----------
    results : dict
        Dictionary with CV results
    """
    print("\n" + "="*80)
    print("SAVING CROSS-VALIDATION RESULTS")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    summary_data = []
    for model_name, res in results.items():
        summary_data.append({
            'Model': model_name,
            'Test_Accuracy': res['test_acc'],
            'Test_Accuracy_Std': res['test_acc_std'],
            'Train_Accuracy': res['train_acc'],
            'Test_F1': res['test_f1'],
            'Test_F1_Std': res['test_f1_std'],
            'Test_Precision': res['test_precision'],
            'Test_Recall': res['test_recall'],
            'Test_ROC_AUC': res['test_auc'],
            'Overfit_Gap': res['overfit_gap']
        })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = f"{RESULTS_DIR}cv_results_10fold.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")


def generate_final_report(results):
    """
    Generate final report for research paper
    
    Parameters:
    -----------
    results : dict
        Dictionary with CV results
    """
    print("\n" + "="*80)
    print("FINAL REPORT FOR RESEARCH PAPER")
    print("="*80)
    
    print(f"\n📊 Dataset: UCI Chronic Kidney Disease")
    print(f"   Methodology: 10-Fold Stratified Cross-Validation")
    print(f"   Class Balancing: SMOTE")
    
    print(f"\n📝 Model Performance:")
    
    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    
    for model_name, res in sorted_results:
        print(f"\n   {model_name}:")
        print(f"      Accuracy:  {res['test_acc']:.4f} ± {res['test_acc_std']:.4f}")
        print(f"      Precision: {res['test_precision']:.4f}")
        print(f"      Recall:    {res['test_recall']:.4f}")
        print(f"      F1-Score:  {res['test_f1']:.4f} ± {res['test_f1_std']:.4f}")
        print(f"      ROC-AUC:   {res['test_auc']:.4f}")
    
    best_model = sorted_results[0]
    print(f"\n🏆 Recommended Model for Paper: {best_model[0]}")
    print(f"   F1-Score: {best_model[1]['test_f1']:.4f} ± {best_model[1]['test_f1_std']:.4f}")
    
    print(f"\n💡 Paper Recommendations:")
    print(f"""
1. Report 10-fold cross-validation results with mean ± std
2. Highlight SMOTE usage for class imbalance
3. Emphasize {best_model[0]} as best performing model
4. Include confusion matrix and ROC curve for best model
5. Discuss clinical interpretability of selected model
""")


def main():
    """
    Main execution function
    """
    print("="*80)
    print("UCI CHRONIC KIDNEY DISEASE - CV TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Load and prepare data
    X, y, label_encoder = load_and_prepare_data(use_cleaned=True)
    
    # Step 2: Prepare features for CV
    X_prepared = prepare_features_for_cv(X)
    
    # Step 3: Perform cross-validation
    cv_results = perform_cross_validation(X_prepared, y)
    
    # Step 4: Train final models on full dataset
    trained_models = train_final_models(X_prepared, y)
    
    # Step 5: Save models
    save_models(trained_models)
    
    # Step 6: Save CV results
    save_cv_results(cv_results)
    
    # Step 7: Generate final report
    generate_final_report(cv_results)
    
    print("\n" + "="*80)
    print("✅ CROSS-VALIDATION TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Models saved to: {MODELS_DIR}")
    print(f"📊 Results saved to: {RESULTS_DIR}")
    
    return cv_results, trained_models


if __name__ == "__main__":
    cv_results, trained_models = main()