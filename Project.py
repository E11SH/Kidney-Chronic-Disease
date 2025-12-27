import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import load_data, identify_target_column, identify_column_types, get_data_info
from data_cleaning import full_cleaning_pipeline, save_cleaned_data
import eda
import visualization

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to orchestrate the entire analysis"""
    
    print("="*80)
    print("KIDNEY DISEASE CLASSIFICATION - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Get data information
    data_info = get_data_info(df)
    target_col = data_info['target_column']
    numerical_cols = data_info['numerical_columns']
    categorical_cols = data_info['categorical_columns']
    
    print(f"\n📊 Data Info:")
    print(f"  - Total samples: {data_info['n_rows']}")
    print(f"  - Total features: {data_info['n_cols']}")
    print(f"  - Numerical features: {data_info['n_numerical']}")
    print(f"  - Categorical features: {data_info['n_categorical']}")
    if target_col:
        print(f"  - Target column: '{target_col}'")
    
    # =========================================================================
    # PHASE 2: INITIAL EDA (BEFORE CLEANING)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: INITIAL EDA (BEFORE CLEANING)")
    print("="*80)
    
    # Display first few rows
    print("\n--- Data Sample ---")
    print(df.head())
    
    # Basic statistics
    eda.get_basic_statistics(df)
    
    # Check uniqueness
    eda.check_uniqueness(df)
    
    # Check target distribution
    eda.check_target_distribution(df, target_col)
    
    # Check categorical values
    eda.check_categorical_values(df, categorical_cols)
    
    # =========================================================================
    # PHASE 3: INITIAL VISUALIZATIONS (BEFORE CLEANING)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: INITIAL VISUALIZATIONS (BEFORE CLEANING)")
    print("="*80)
    
    # Numerical distributions
    visualization.plot_numerical_distributions(df, numerical_cols)
    
    # Box plots
    visualization.plot_boxplots(df, numerical_cols)
    
    # Categorical distributions
    visualization.plot_categorical_distributions(df, categorical_cols)
    
    # Correlation heatmap
    visualization.plot_correlation_heatmap(df, numerical_cols)
    
    # Missing values
    visualization.plot_missing_values(df)
    
    # Duplicates
    visualization.plot_duplicates(df)
    
    # Outliers
    visualization.plot_outliers(df, numerical_cols)
    
    # =========================================================================
    # PHASE 4: DATA CLEANING
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: DATA CLEANING")
    print("="*80)
    
    # Execute full cleaning pipeline
    df_clean, cleaning_report = full_cleaning_pipeline(df, target_col)
    
    # Update column lists after cleaning
    numerical_cols_clean = df_clean.select_dtypes(include=['number']).columns.tolist()
    categorical_cols_clean = df_clean.select_dtypes(include='object').columns.tolist()
    
    # =========================================================================
    # PHASE 5: ADVANCED EDA (AFTER CLEANING)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 5: ADVANCED EDA (AFTER CLEANING)")
    print("="*80)
    
    # Skewness and Kurtosis
    skew_kurt_df = eda.check_skewness_kurtosis(df_clean, numerical_cols_clean)
    
    # Apply transformations
    df_transformed, transformation_info = eda.apply_transformations(
        df_clean, numerical_cols_clean, skew_kurt_df
    )
    
    # VIF for multicollinearity
    vif_df = eda.calculate_vif(df_clean, numerical_cols_clean)
    
    # Logical validity check
    validity_issues = eda.check_logical_validity(df_clean, numerical_cols_clean)
    
    # Rare categories
    rare_categories = eda.identify_rare_categories(df_clean, categorical_cols_clean)
    
    # GroupBy analysis
    groupby_results = eda.perform_groupby_analysis(
        df_clean, categorical_cols_clean, numerical_cols_clean, target_col
    )
    
    # PCA
    if len(numerical_cols_clean) >= 2:
        pca_result = eda.perform_pca(df_clean, numerical_cols_clean, n_components=2)
    else:
        pca_result = None
    
    # Clustering
    if len(numerical_cols_clean) >= 2:
        cluster_result = eda.perform_clustering(df_clean, numerical_cols_clean, n_clusters=3)
    else:
        cluster_result = None
    
    # =========================================================================
    # PHASE 6: ADVANCED VISUALIZATIONS (AFTER CLEANING)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 6: ADVANCED VISUALIZATIONS (AFTER CLEANING)")
    print("="*80)
    
    # Skewness/Kurtosis visualization
    visualization.plot_skewness_kurtosis(skew_kurt_df)
    
    # Scatter matrix
    visualization.plot_scatter_matrix(df_clean, numerical_cols_clean)
    
    # Categorical vs Numerical
    if len(categorical_cols_clean) > 0 and len(numerical_cols_clean) > 0:
        visualization.plot_categorical_vs_numerical(
            df_clean, categorical_cols_clean, numerical_cols_clean
        )
    
    # Categorical vs Target
    if target_col and len(categorical_cols_clean) > 0:
        visualization.plot_categorical_vs_target(
            df_clean, categorical_cols_clean, target_col
        )
    
    # Pairplot
    visualization.plot_pairplot(df_clean, numerical_cols_clean, target_col)
    
    # 3D Scatter
    if len(numerical_cols_clean) >= 3:
        visualization.plot_3d_scatter(df_clean, numerical_cols_clean, target_col)
    
    # PCA visualization
    if pca_result is not None:
        visualization.plot_pca_results(pca_result, target_col, df_clean)
    
    # Cluster visualization
    if cluster_result is not None and pca_result is not None:
        visualization.plot_clusters(cluster_result, pca_result)
    
    # =========================================================================
    # PHASE 7: FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 7: FINAL SUMMARY")
    print("="*80)
    
    # Generate comprehensive EDA summary
    eda.generate_eda_summary(df_clean, numerical_cols_clean, categorical_cols_clean, target_col)
    
    # Save cleaned data
    save_cleaned_data(df_clean)
    
    print("\n" + "="*80)
    print("✅ EDA & PREPROCESSING COMPLETE!")
    print("="*80)
    print("\n📁 Files Generated:")
    print("  - College/kidney_disease_cleaned.csv (cleaned dataset)")
    print("  - College/diagrams/ (all visualization PNG files)")
    
    # =========================================================================
    # PHASE 8: MODEL TRAINING (NEW!)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 8: MACHINE LEARNING MODEL TRAINING")
    print("="*80)
    
    user_input = input("\nDo you want to train the ML models now? (y/n): ").lower()
    
    if user_input == 'y':
        print("\n🚀 Starting cross-validation training pipeline...")
        from train_models import main as train_cv
        cv_results, trained_models = train_cv()
    else:
        print("\n⏭️  Model training skipped. Run 'train_models.py' manually when ready.")
        print("\n🎉 EDA phase completed successfully!")
    
    return df_clean, cleaning_report


# =============================================================================
# RUN THE ANALYSIS
# =============================================================================
if __name__ == "__main__":
    df_clean, report = main()
    