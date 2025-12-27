# models/traditional_models_base.py

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib  # Using joblib for saving the pipeline

# SMOTE imports
from imblearn.over_sampling import SMOTE
import warnings

# Suppress specific FutureWarning from sklearn/imblearn interaction
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*BaseEstimator._validate_data.*")

class TraditionalCKDClassifier:
    """Base class for traditional scikit-learn classifiers with SMOTE support."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.pipeline = None
        self.X_train_processed = None
        self.y_train_processed = None
        # Target classes
        # Target classes for binary classification (0/1)
        self.class_mapping = {
            0: 0, 
            1: 1
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.target_names = ['No Disease', 'CKD'] # Explicit names for reporting


    def _inject_noise(self, X, noise_level):
        """
        Inject Gaussian noise into numerical features for data augmentation.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        noise_level : float
            Standard deviation of noise as a fraction of feature std
            
        Returns:
        --------
        X_noisy : pandas.DataFrame
            Features with added noise
        """
        if noise_level <= 0:
            return X
        
        X_noisy = X.copy()
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        
        for col in numerical_features:
            # Calculate noise std as fraction of feature std
            feature_std = X[col].std()
            noise_std = noise_level * feature_std
            
            # Generate and add Gaussian noise
            noise = np.random.normal(0, noise_std, size=len(X))
            X_noisy[col] = X[col] + noise
        
        return X_noisy

    def _prepare_data(self, df, target_col, test_size):
        """Prepares data, defines preprocessor, and splits into train/test sets."""
        
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert target to numerical labels
        y = y.map(self.class_mapping)

        # Identify column types
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()

        # Define preprocessor: Scale numerical, One-hot encode categorical
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        # Split data (using stratify to maintain class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test

    def train(self, df, target_col, test_size=0.2, verbose=True, use_smote=True, noise_level=0.0):
        """Trains the model and evaluates performance with optional SMOTE and noise injection.
        
        Parameters:
        -----------
        noise_level : float
            Standard deviation of Gaussian noise as fraction of feature std (default: 0.0)
            Recommended values: 0.01-0.1 for regularization
        """
        
        X_train, X_test, y_train, y_test = self._prepare_data(df, target_col, test_size)
        
        # Apply noise injection to training data BEFORE preprocessing
        if noise_level > 0:
            if verbose:
                print(f"\n✓ Applying Gaussian noise injection (level={noise_level})...")
            X_train = self._inject_noise(X_train, noise_level)
        
        if verbose:
            print(f"\nOriginal training set size: {len(y_train)}")
            print(f"Class distribution before SMOTE:")
            train_dist = pd.Series(y_train).value_counts().sort_index()
            for class_idx, count in train_dist.items():
                class_name = self.reverse_class_mapping.get(class_idx, "Unknown")
                print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        start_time = time.time()
        
        # FIT preprocessor on training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Apply SMOTE AFTER preprocessing
        if use_smote:
            if verbose:
                print(f"\n✓ Applying SMOTE to balance classes...")
            
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            
            if verbose:
                print(f"Resampled training set size: {len(y_train_resampled)}")
                print(f"Class distribution after SMOTE:")
                resampled_dist = pd.Series(y_train_resampled).value_counts().sort_index()
                for class_idx, count in resampled_dist.items():
                    class_name = self.reverse_class_mapping.get(class_idx, "Unknown")
                    print(f"  {class_name}: {count} ({count/len(y_train_resampled)*100:.1f}%)")
        else:
            if verbose:
                print(f"\n⚠️ Training WITHOUT SMOTE (class imbalance not addressed)...")
            X_train_resampled = X_train_processed
            y_train_resampled = y_train
        
        # Train the classifier on resampled data
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Create a simple pipeline for prediction (preprocessor + trained model)
        # We can't use Pipeline here because model is already trained
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor)
        ])
        # Store trained model separately
        self.trained_classifier = self.model
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_processed)
        
        # Print prediction distribution
        if verbose:
            print(f"\nPrediction distribution on test set:")
            pred_series = pd.Series(y_pred)
            pred_dist = pred_series.value_counts().sort_index()
            
            # Check if all classes are predicted
            all_classes_predicted = len(pred_dist) == len(self.target_names)
            
            for class_idx in range(2):
                class_name = self.target_names[class_idx]
                pred_count = pred_dist.get(class_idx, 0)
                actual_count = (y_test == class_idx).sum()
                print(f"  {class_name}: Predicted={pred_count} ({pred_count/len(y_pred)*100:.1f}%), Actual={actual_count}")
            
            if not all_classes_predicted:
                print(f"\n⚠️ WARNING: Model is not predicting all classes!")
        
        # Metrics calculation
        train_pred = self.model.predict(X_train_resampled)
        training_accuracy = accuracy_score(y_train_resampled, train_pred)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        
        # Classification report
        report_dict = classification_report(
            y_test, y_pred, 
            target_names=self.target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'training_accuracy': training_accuracy,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict,
            'training_time': training_time,
            'used_smote': use_smote,
            'noise_level': noise_level
        }

        if verbose:
            print(f"\nModel Training Complete in {training_time:.2f}s.")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            
        return results

    def predict(self, input_df):
        """Makes a prediction for a single input DataFrame and returns class label."""
        if self.trained_classifier is None:
            raise Exception("Model not trained or loaded.")
        
        # Preprocess input
        input_processed = self.preprocessor.transform(input_df)
        
        # Predict
        pred_label = self.trained_classifier.predict(input_processed)[0]
        return self.reverse_class_mapping.get(pred_label, "Unknown")
    
    def predict_proba(self, input_df):
        """Returns the class probabilities and class names."""
        if self.trained_classifier is None:
            raise Exception("Model not trained or loaded.")
        
        # Preprocess input
        input_processed = self.preprocessor.transform(input_df)
        
        # Get probabilities
        probabilities = self.trained_classifier.predict_proba(input_processed)
        
        return probabilities, self.target_names
    
    def save_model(self, filepath):
        """Saves the preprocessor and trained classifier."""
        if self.trained_classifier is None:
            print(f"⚠️ Cannot save {self.__class__.__name__}: Model is not trained.")
            return
        
        # Save both preprocessor and classifier
        model_data = {
            'preprocessor': self.preprocessor,
            'classifier': self.trained_classifier,
            'class_mapping': self.class_mapping,
            'reverse_class_mapping': self.reverse_class_mapping,
            'target_names': self.target_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ {self.__class__.__name__} saved to '{filepath}'")

    def load_model(self, filepath):
        """Loads the preprocessor and trained classifier from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.preprocessor = model_data['preprocessor']
            self.trained_classifier = model_data['classifier']
            self.class_mapping = model_data['class_mapping']
            self.reverse_class_mapping = model_data['reverse_class_mapping']
            self.target_names = model_data['target_names']
            
            # Recreate pipeline for compatibility
            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor)
            ])
            
            print(f"✓ {self.__class__.__name__} loaded from '{filepath}'")
            return True
        except FileNotFoundError:
            print(f"❌ ERROR: Model file '{filepath}' not found.")
            return False
        except Exception as e:
            print(f"❌ ERROR loading {self.__class__.__name__}: {str(e)}")
            return False