"""
Comprehensive Dataset Leakage & Quality Checker
================================================
Run this FIRST on any new dataset to detect problems before training
Usage: Place your dataset as 'kidney.csv' and run this script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DatasetHealthChecker:
    """Comprehensive checker for ML dataset quality and leakage."""
    
    def __init__(self, filepath, target_col=None):
        """
        Initialize checker.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        target_col : str or None
            Target column name. If None, will try to auto-detect
        """
        self.filepath = filepath
        self.df = None
        self.target_col = target_col
        self.report = {}
        
    def load_and_inspect(self):
        """Load data and perform initial inspection."""
        print("="*80)
        print("DATASET LOADING AND INITIAL INSPECTION")
        print("="*80)
        
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✓ Successfully loaded: {self.filepath}")
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {self.filepath}")
            return False
        except Exception as e:
            print(f"❌ ERROR loading file: {e}")
            return False
        
        print(f"\n📊 Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\n📋 Columns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            nulls = self.df[col].isnull().sum()
            unique = self.df[col].nunique()
            print(f"  {i:2d}. {col:<30} | Type: {str(dtype):<10} | Nulls: {nulls:>4} | Unique: {unique:>4}")
        
        # Auto-detect target column if not specified
        if self.target_col is None:
            self.target_col = self._auto_detect_target()
        
        if self.target_col:
            print(f"\n🎯 Target Column: '{self.target_col}'")
            print(f"   Class Distribution:")
            dist = self.df[self.target_col].value_counts().sort_index()
            for class_val, count in dist.items():
                pct = count / len(self.df) * 100
                print(f"   - {class_val}: {count} ({pct:.1f}%)")
        else:
            print(f"\n⚠️ WARNING: Target column not specified or detected!")
            return False
        
        self.report['shape'] = self.df.shape
        self.report['target'] = self.target_col
        self.report['class_distribution'] = dist.to_dict()
        
        return True
    
    def _auto_detect_target(self):
        """Try to auto-detect target column based on common naming patterns."""
        target_patterns = [
            'target', 'label', 'class', 'classification', 'diagnosis',
            'disease', 'outcome', 'status', 'ckd', 'result'
        ]
        
        for col in self.df.columns:
            col_lower = col.lower()
            for pattern in target_patterns:
                if pattern in col_lower:
                    print(f"\n🔍 Auto-detected target column: '{col}'")
                    return col
        
        # If not found, look for binary columns (likely target)
        for col in self.df.columns:
            if self.df[col].nunique() == 2:
                print(f"\n🔍 Auto-detected binary target column: '{col}'")
                return col
        
        return None
    
    def check_missing_data(self):
        """Check for missing values and data quality issues."""
        print("\n" + "="*80)
        print("MISSING DATA ANALYSIS")
        print("="*80)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\n⚠️ Found {len(missing_df)} columns with missing values:")
            print(missing_df.to_string(index=False))
            
            print(f"\n💡 Recommendation:")
            for _, row in missing_df.iterrows():
                if row['Percentage'] > 50:
                    print(f"   - {row['Column']}: {row['Percentage']}% missing → Consider dropping column")
                elif row['Percentage'] > 20:
                    print(f"   - {row['Column']}: {row['Percentage']}% missing → Use careful imputation")
                else:
                    print(f"   - {row['Column']}: {row['Percentage']}% missing → Simple imputation OK")
        else:
            print("\n✓ No missing values detected!")
        
        self.report['missing_data'] = missing_df.to_dict('records') if len(missing_df) > 0 else None
        
    def check_data_leakage(self, threshold=0.7):
        """
        Check for data leakage by examining correlations with target.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold above which to flag potential leakage (default: 0.7)
        """
        print("\n" + "="*80)
        print("DATA LEAKAGE DETECTION")
        print("="*80)
        
        if self.target_col not in self.df.columns:
            print("❌ Target column not found. Cannot check for leakage.")
            return
        
        # Separate numerical and categorical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != self.target_col]
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != self.target_col]
        
        leaky_features = []
        correlations = {}
        
        # Check numerical features
        if len(numerical_cols) > 0:
            print(f"\n📊 NUMERICAL FEATURES ({len(numerical_cols)}):")
            print(f"{'Feature':<30} {'Correlation':<15} {'P-value':<12} {'Status'}")
            print("-" * 80)
            
            for col in numerical_cols:
                try:
                    # Handle missing values
                    valid_mask = ~(self.df[col].isnull() | self.df[self.target_col].isnull())
                    if valid_mask.sum() < 10:
                        print(f"{col:<30} {'N/A':<15} {'N/A':<12} ⚠️ Insufficient data")
                        continue
                    
                    corr, p_value = pearsonr(
                        self.df.loc[valid_mask, col],
                        self.df.loc[valid_mask, self.target_col]
                    )
                    
                    abs_corr = abs(corr)
                    correlations[col] = corr
                    
                    if abs_corr >= threshold:
                        status = "🚨 LEAKAGE!"
                        leaky_features.append((col, abs_corr, 'numerical'))
                    elif abs_corr >= 0.5:
                        status = "⚠️ Suspicious"
                    else:
                        status = "✓ OK"
                    
                    print(f"{col:<30} {corr:>+.6f}      {p_value:<12.6f} {status}")
                
                except Exception as e:
                    print(f"{col:<30} Error: {str(e)[:30]}")
        
        # Check categorical features using Cramér's V
        if len(categorical_cols) > 0:
            print(f"\n📊 CATEGORICAL FEATURES ({len(categorical_cols)}):")
            cramers_v_label = "Cramér's V"
            print(f"{'Feature':<30} {cramers_v_label:<15} {'Status'}")
            print("-" * 70)
            
            for col in categorical_cols:
                try:
                    # Create contingency table
                    contingency = pd.crosstab(self.df[col], self.df[self.target_col])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    # Calculate Cramér's V
                    n = contingency.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                    
                    correlations[col] = cramers_v
                    
                    if cramers_v >= threshold:
                        status = "🚨 LEAKAGE!"
                        leaky_features.append((col, cramers_v, 'categorical'))
                    elif cramers_v >= 0.5:
                        status = "⚠️ Suspicious"
                    else:
                        status = "✓ OK"
                    
                    print(f"{col:<30} {cramers_v:>+.6f}      {status}")
                
                except Exception as e:
                    print(f"{col:<30} Error: {str(e)[:30]}")
        
        # Summary
        if leaky_features:
            print(f"\n{'='*80}")
            print(f"⚠️ WARNING: Found {len(leaky_features)} potentially leaky feature(s)!")
            print(f"{'='*80}")
            for feat, corr, feat_type in leaky_features:
                print(f"  • {feat} ({feat_type}): {abs(corr):.3f}")
            print(f"\n💡 These features may directly encode the target variable.")
            print(f"   Consider removing them or investigating their relationship.")
        else:
            print(f"\n✓ No obvious data leakage detected (threshold={threshold})")
        
        self.report['leaky_features'] = leaky_features
        self.report['correlations'] = correlations
        
        return correlations, leaky_features
    
    def check_feature_importance(self, top_n=10):
        """Use Random Forest to identify most important features."""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Prepare data
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Handle categorical variables
        X_encoded = X.copy()
        le_dict = {}
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            le_dict[col] = le
        
        # Fill missing values with median/mode
        for col in X_encoded.columns:
            if X_encoded[col].isnull().any():
                if X_encoded[col].dtype in [np.float64, np.int64]:
                    X_encoded[col].fillna(X_encoded[col].median(), inplace=True)
                else:
                    X_encoded[col].fillna(X_encoded[col].mode()[0], inplace=True)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest to assess feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_encoded, y)
        
        # Get feature importance
        importances = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📊 Top {top_n} Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
        print("-" * 50)
        for i, row in importances.head(top_n).iterrows():
            print(f"{i+1:<6} {row['Feature']:<30} {row['Importance']:.6f}")
        
        self.report['feature_importance'] = importances.to_dict('records')
        
        return importances
    
    def check_class_balance(self):
        """Check if dataset is balanced."""
        print("\n" + "="*80)
        print("CLASS BALANCE ANALYSIS")
        print("="*80)
        
        dist = self.df[self.target_col].value_counts()
        total = len(self.df)
        
        print(f"\nClass distribution:")
        for class_val, count in dist.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {class_val}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        # Check balance
        imbalance_ratio = dist.max() / dist.min()
        
        if imbalance_ratio > 3:
            print(f"\n⚠️ SEVERE IMBALANCE detected! (ratio: {imbalance_ratio:.1f}:1)")
            print(f"💡 Recommendation: Use SMOTE, class weights, or stratified sampling")
        elif imbalance_ratio > 1.5:
            print(f"\n⚠️ Moderate imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print(f"💡 Recommendation: Consider using class weights or SMOTE")
        else:
            print(f"\n✓ Classes are reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
        
        self.report['imbalance_ratio'] = imbalance_ratio
    
    def check_sample_size(self):
        """Assess if sample size is adequate."""
        print("\n" + "="*80)
        print("SAMPLE SIZE ASSESSMENT")
        print("="*80)
        
        n_samples = len(self.df)
        n_features = len(self.df.columns) - 1  # Exclude target
        samples_per_feature = n_samples / n_features
        
        print(f"\nSamples: {n_samples}")
        print(f"Features: {n_features}")
        print(f"Samples per feature: {samples_per_feature:.1f}")
        
        # Rules of thumb
        if samples_per_feature < 10:
            print(f"\n🚨 CRITICAL: Very small sample size!")
            print(f"   Risk: High overfitting, unreliable results")
            print(f"   Recommendations:")
            print(f"   1. Use simpler models (Logistic Regression, small trees)")
            print(f"   2. Apply strong regularization")
            print(f"   3. Use nested cross-validation")
            print(f"   4. Report confidence intervals")
        elif samples_per_feature < 50:
            print(f"\n⚠️ WARNING: Small dataset")
            print(f"   Recommendations:")
            print(f"   1. Use cross-validation (k=5 or k=10)")
            print(f"   2. Feature selection to reduce dimensionality")
            print(f"   3. Regularization (L1/L2)")
            print(f"   4. Report std deviations")
        else:
            print(f"\n✓ Sample size appears adequate")
            print(f"   Standard ML practices apply")
        
        self.report['sample_size'] = n_samples
        self.report['samples_per_feature'] = samples_per_feature
    
    def visualize_leakage(self, correlations, leaky_features, output_path='leakage_visualization.png'):
        """Create visualizations of potential data leakage."""
        if not correlations:
            print("\nNo correlations to visualize.")
            return
        
        # Filter to only numerical correlations for visualization
        num_correlations = {k: v for k, v in correlations.items() 
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
        
        if not num_correlations:
            print("\nNo numerical correlations to visualize.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Correlation bar chart
        ax1 = axes[0]
        sorted_corr = dict(sorted(num_correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        features = list(sorted_corr.keys())[:15]  # Top 15
        values = [sorted_corr[f] for f in features]
        colors = ['red' if abs(v) >= 0.7 else 'orange' if abs(v) >= 0.5 else 'green' for v in values]
        
        ax1.barh(features, values, color=colors, alpha=0.7)
        ax1.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Leakage threshold')
        ax1.axvline(x=-0.7, color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Correlation with Target', fontsize=12)
        ax1.set_title('Feature-Target Correlations\n(Red = Potential Leakage)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Summary statistics
        ax2 = axes[1]
        ax2.axis('off')
        
        summary_text = "DATASET HEALTH SUMMARY\n" + "="*40 + "\n\n"
        summary_text += f"Total Samples: {self.report['shape'][0]}\n"
        summary_text += f"Total Features: {self.report['shape'][1] - 1}\n"
        summary_text += f"Samples/Feature: {self.report.get('samples_per_feature', 0):.1f}\n\n"
        
        summary_text += f"Class Balance:\n"
        for class_val, count in self.report['class_distribution'].items():
            pct = count / self.report['shape'][0] * 100
            summary_text += f"  {class_val}: {count} ({pct:.1f}%)\n"
        
        summary_text += f"\nImbalance Ratio: {self.report.get('imbalance_ratio', 0):.1f}:1\n\n"
        
        if leaky_features:
            summary_text += f"⚠️ LEAKY FEATURES ({len(leaky_features)}):\n"
            for feat, corr, ftype in leaky_features[:5]:
                summary_text += f"  • {feat[:25]}: {abs(corr):.3f}\n"
        else:
            summary_text += "✓ No data leakage detected\n"
        
        ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
    
    def generate_full_report(self):
        """Run all checks and generate comprehensive report."""
        if not self.load_and_inspect():
            return None
        
        self.check_missing_data()
        self.check_sample_size()
        self.check_class_balance()
        correlations, leaky_features = self.check_data_leakage(threshold=0.7)
        importances = self.check_feature_importance()
        
        # Visualize
        self.visualize_leakage(correlations, leaky_features, 'College/kidney_leakage_check.png')
        
        # Final recommendations
        print("\n" + "="*80)
        print("FINAL RECOMMENDATIONS")
        print("="*80)
        
        issues = []
        if self.report.get('samples_per_feature', float('inf')) < 50:
            issues.append("Small sample size")
        if self.report.get('imbalance_ratio', 1) > 2:
            issues.append("Class imbalance")
        if leaky_features:
            issues.append(f"{len(leaky_features)} leaky features")
        if self.report.get('missing_data'):
            issues.append("Missing values present")
        
        if issues:
            print(f"\n⚠️ ISSUES DETECTED: {', '.join(issues)}")
            print(f"\nRECOMMENDED ACTIONS:")
            print(f"1. Address leaky features (remove or investigate)")
            print(f"2. Use 10-fold stratified cross-validation")
            print(f"3. Apply SMOTE if imbalanced")
            print(f"4. Use regularized models (avoid overfitting)")
            print(f"5. Report confidence intervals/std deviations")
        else:
            print(f"\n✓ Dataset looks healthy! Proceed with standard ML workflow.")
        
        print(f"\n{'='*80}")
        print(f"✅ COMPREHENSIVE HEALTH CHECK COMPLETE!")
        print(f"{'='*80}")
        
        return self.report


# ============================================================================
# USAGE - UCI CKD Dataset
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("UCI CHRONIC KIDNEY DISEASE DATASET - HEALTH CHECK")
    print("="*80)
    print("\nOption 1: Load from ucimlrepo (recommended)")
    print("Option 2: Load from local CSV file")
    
    try:
        # Try loading from ucimlrepo package
        from ucimlrepo import fetch_ucirepo
        
        print("\n📥 Fetching dataset from UCI ML Repository...")
        chronic_kidney_disease = fetch_ucirepo(id=336)
        
        # Combine features and target
        X = chronic_kidney_disease.data.features
        y = chronic_kidney_disease.data.targets
        
        df = pd.concat([X, y], axis=1)
        
        # Save locally
        df.to_csv('College/kidney.csv', index=False)
        print(f"✓ Dataset saved to: College/kidney.csv")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Run checker
        # Target is usually 'class' or 'classification' in UCI datasets
        target_col = y.columns[0] if hasattr(y, 'columns') else 'class'
        checker = DatasetHealthChecker('College/kidney.csv', target_col=target_col)
        report = checker.generate_full_report()
        
    except ImportError:
        print("\n⚠️ ucimlrepo package not found. Install with:")
        print("   pip install ucimlrepo")
        print("\nOr place your kidney.csv file in College/ folder and run again.")
        
        # Try loading from local file
        try:
            checker = DatasetHealthChecker('College/kidney.csv', target_col=None)
            report = checker.generate_full_report()
        except FileNotFoundError:
            print("\n❌ kidney.csv not found in College/ folder")
            print("   Please download the dataset or install ucimlrepo")
    
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nTrying local file...")
        try:
            checker = DatasetHealthChecker('College/kidney.csv', target_col=None)
            report = checker.generate_full_report()
        except FileNotFoundError:
            print("\n❌ kidney.csv not found in College/ folder")
    
    print(f"\n💾 Full report saved in 'report' variable")
    print(f"📊 Visualization saved to: College/kidney_leakage_check.png")