"""
Data Cleaning Module
Contains all data cleaning and preprocessing functions
"""

import pandas as pd
import numpy as np
from config import MISSING_VALUE_INDICATORS, IQR_MULTIPLIER, CLEANED_DATA_PATH

# ... (existing imports, skipping to save_cleaned_data)

def remove_id_column(df):
    """Remove ID column if present"""
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        print("✓ Removed 'id' column.")
    return df


def clean_string_values(df):
    """Remove whitespace and tabs from string columns"""
    print("\n--- Cleaning String Values (Whitespace/Tabs) ---")
    categorical_cols = df.select_dtypes(include='object').columns
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    print(f"✓ Cleaned {len(categorical_cols)} string columns.")
    return df


def convert_to_numeric(df, target_col=None):
    """Convert string columns to numeric where appropriate"""
    print("\n--- Converting String Columns to Numeric ---")
    converted_cols = 0
    
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_col:
            # Replace missing value indicators with NaN
            test_numeric = pd.to_numeric(
                df[col].replace(MISSING_VALUE_INDICATORS, np.nan), 
                errors='coerce'
            )
            
            # Check if conversion resulted in numeric values
            if test_numeric.notna().sum() > 0 and test_numeric.dtype != 'object':
                df[col] = pd.to_numeric(
                    df[col].replace(MISSING_VALUE_INDICATORS, np.nan), 
                    errors='coerce'
                )
                print(f"✓ Converted '{col}' to numeric (now {df[col].dtype}).")
                converted_cols += 1
    
    print(f"Summary: Converted {converted_cols} columns to numeric.")
    return df


def assess_missing_values(df):
    """Assess and report missing values"""
    print("\n--- Missing Values Assessment ---")
    
    missing_values = df.isnull().sum()
    missing_summary = missing_values[missing_values > 0].sort_values(ascending=False)
    total_missing_count = missing_values.sum()
    
    if total_missing_count > 0:
        print(f"\nTotal missing values: {total_missing_count}")
        print("\nMissing values per column:")
        print(missing_summary.to_frame('Missing Count'))
    else:
        print("✓ No missing values found!")
    
    return missing_summary


def remove_duplicates(df):
    """Detect and remove duplicate rows"""
    print("\n--- Duplicate Rows Detection & Removal ---")
    
    duplicates_before = df.duplicated().sum()
    print(f"Number of duplicate rows found: {duplicates_before}")
    
    if duplicates_before > 0:
        df = df.drop_duplicates()
        print(f"✓ Dropped {duplicates_before} duplicate rows. New shape: {df.shape}")
    else:
        print("✓ No duplicate rows found.")
    
    return df


def detect_outliers(df, numerical_cols):
    """
    Detect outliers using IQR method
    
    Returns:
    --------
    outlier_info : dict
        Dictionary with outlier counts for each column
    """
    print("\n--- Outlier Detection (IQR Method) ---")
    
    outlier_info = {}
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = len(outliers)
        
        print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    
    total_outliers = sum(outlier_info.values())
    print(f"⚠️ Note: Outliers detected ({total_outliers} total) but NOT removed (common for medical data).")
    
    return outlier_info


def impute_missing_values(df, target_col=None):
    """
    Impute missing values
    - Numerical: Median
    - Categorical: Mode
    """
    print("\n--- Imputing Missing Values (Median/Mode) ---")
    
    if df.isnull().sum().sum() > 0:
        print("Strategy: Numerical -> Median, Categorical -> Mode.")
        
        if 'Medication' in df.columns:
            df['Medication'].fillna('No_Medication', inplace=True)
            print("✓ Imputed 'Medication' missing values with 'No_Medication'.")

        # Impute numerical columns with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Impute categorical columns with mode
        for col in df.select_dtypes(include='object').columns:
            if col != target_col and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"✓ All missing values imputed. Final missing values: {df.isnull().sum().sum()}")
    else:
        print("✓ No missing values to impute!")
    
    return df


def full_cleaning_pipeline(df, target_col=None):
    """
    Execute full cleaning pipeline
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    df_clean : pandas.DataFrame
        Cleaned dataset
    cleaning_report : dict
        Report of cleaning operations
    """
    print("\n" + "="*80)
    print("EXECUTING DATA CLEANING PIPELINE")
    print("="*80)
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    print(f"Initial shape: {initial_shape}")
    
    # Step 1: Remove ID column
    df_clean = remove_id_column(df_clean)
    
    # Step 2: Clean string values
    df_clean = clean_string_values(df_clean)
    
    # Step 3: Convert to numeric
    df_clean = convert_to_numeric(df_clean, target_col)
    
    # Step 4: Assess missing values
    missing_summary = assess_missing_values(df_clean)
    
    # Step 5: Remove duplicates
    df_clean = remove_duplicates(df_clean)
    
    # Step 6: Detect outliers (but don't remove)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = detect_outliers(df_clean, numerical_cols)
    
    # Step 7: Impute missing values
    df_clean = impute_missing_values(df_clean, target_col)
    
    # Create cleaning report
    cleaning_report = {
        'initial_shape': initial_shape,
        'final_shape': df_clean.shape,
        'rows_removed': initial_shape[0] - df_clean.shape[0],
        'missing_values_final': df_clean.isnull().sum().sum(),
        'outliers_detected': outlier_info,
        'data_quality': 'CLEANED' if df_clean.isnull().sum().sum() == 0 else 'REQUIRES ATTENTION'
    }
    
    print("\n" + "="*80)
    print("CLEANING PIPELINE COMPLETE")
    print("="*80)
    print(f"Original dataset size: {initial_shape}")
    print(f"Cleaned dataset size: {df_clean.shape}")
    print(f"Rows removed (duplicates): {cleaning_report['rows_removed']}")
    print(f"Data Quality Check: {cleaning_report['data_quality']}")
    
    return df_clean, cleaning_report


def save_cleaned_data(df, filepath=CLEANED_DATA_PATH):
    """Save cleaned dataset to CSV"""
    print("\n💾 Saving cleaned dataset...")
    # Ensure the directory exists
    import os
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved as '{filepath}'")