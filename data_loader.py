"""
Data Loading Module
Handles loading data and identifying column types
"""

import pandas as pd
import numpy as np
from config import RAW_DATA_PATH, CLEANED_DATA_PATH


def load_data(filepath=RAW_DATA_PATH):
    """
    Load the dataset from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: '{filepath}' not found. Check file path.")
        exit()


def identify_target_column(df):
    """
    Automatically identify the target column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
        
    Returns:
    --------
    target_col : str or None
        Name of the target column
    """
    for col in df.columns:
        if 'class' in col.lower() or 'target' in col.lower() or 'status' in col.lower():
            return col
    return None


def identify_column_types(df):
    """
    Identify numerical and categorical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
        
    Returns:
    --------
    numerical_cols : list
        List of numerical column names
    categorical_cols : list
        List of categorical column names
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    return numerical_cols, categorical_cols


def get_data_info(df):
    """
    Get basic information about the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
        
    Returns:
    --------
    info_dict : dict
        Dictionary containing dataset information
    """
    target_col = identify_target_column(df)
    numerical_cols, categorical_cols = identify_column_types(df)
    
    info_dict = {
        'shape': df.shape,
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'total_cells': df.shape[0] * df.shape[1],
        'target_column': target_col,
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'n_numerical': len(numerical_cols),
        'n_categorical': len(categorical_cols)
    }
    
    return info_dict


