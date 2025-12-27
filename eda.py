"""
Exploratory Data Analysis (EDA) Module
Contains all analysis and statistical functions
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from config import SKEWNESS_THRESHOLD, VIF_THRESHOLD, RARE_CATEGORY_THRESHOLD


def get_basic_statistics(df):
    """Display basic statistical summary"""
    print("\n--- Statistical Summary (Numeric Features) ---")
    print(df.describe().T)


def check_uniqueness(df):
    """Check unique values and balance in dataset"""
    print("\n--- Uniqueness and Balance Check ---")
    print("Number of Unique Values per Column:")
    print(df.nunique().sort_values(ascending=False).to_frame('Unique Count'))


def check_target_distribution(df, target_col):
    """Check target variable distribution"""
    if target_col:
        print(f"\nTarget column: '{target_col}'")
        print(f"Class distribution (%):")
        print(df[target_col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')


def check_categorical_values(df, categorical_cols):
    """Display value counts for categorical features"""
    print("\nCategorical Feature Value Counts:")
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts(dropna=False))


def check_skewness_kurtosis(df, numerical_cols):
    """
    Check skewness and kurtosis for numerical features
    
    Returns:
    --------
    skew_kurt_df : pandas.DataFrame
        DataFrame with skewness and kurtosis values
    """
    print("\n--- Skewness and Kurtosis Analysis ---")
    
    skew_values = []
    kurt_values = []
    
    for col in numerical_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        skew_values.append(skew)
        kurt_values.append(kurt)
        
        skew_interpretation = "Highly Skewed" if abs(skew) > SKEWNESS_THRESHOLD else "Normal"
        print(f"{col}: Skewness = {skew:.3f} ({skew_interpretation}), Kurtosis = {kurt:.3f}")
    
    skew_kurt_df = pd.DataFrame({
        'Feature': numerical_cols,
        'Skewness': skew_values,
        'Kurtosis': kurt_values
    })
    
    return skew_kurt_df


def apply_transformations(df, numerical_cols, skew_kurt_df):
    """
    Apply log and sqrt transformations to highly skewed features
    
    Returns:
    --------
    df_transformed : pandas.DataFrame
        DataFrame with transformed features
    transformation_info : dict
        Information about applied transformations
    """
    print("\n--- Applying Transformations to Skewed Features ---")
    
    df_transformed = df.copy()
    transformation_info = {}
    
    for idx, col in enumerate(numerical_cols):
        skewness = skew_kurt_df.loc[skew_kurt_df['Feature'] == col, 'Skewness'].values[0]
        
        if abs(skewness) > SKEWNESS_THRESHOLD:
            # Only apply if all values are positive
            if df[col].min() > 0:
                df_transformed[f'{col}_log'] = np.log(df[col])
                df_transformed[f'{col}_sqrt'] = np.sqrt(df[col])
                transformation_info[col] = ['log', 'sqrt']
                print(f"✓ Applied log & sqrt transformations to '{col}'")
            elif df[col].min() >= 0:
                df_transformed[f'{col}_sqrt'] = np.sqrt(df[col])
                transformation_info[col] = ['sqrt']
                print(f"✓ Applied sqrt transformation to '{col}'")
    
    return df_transformed, transformation_info


def calculate_vif(df, numerical_cols):
    """
    Calculate Variance Inflation Factor (VIF) to check multicollinearity
    
    Returns:
    --------
    vif_df : pandas.DataFrame
        DataFrame with VIF values
    """
    print("\n--- Multicollinearity Check (VIF) ---")
    
    # Remove any columns with NaN values for VIF calculation
    df_clean = df[numerical_cols].dropna()
    
    if df_clean.shape[1] < 2:
        print("⚠️ Not enough features for VIF calculation")
        return None
    
    vif_data = []
    
    for i, col in enumerate(df_clean.columns):
        try:
            # Get all other columns except the current one
            X = df_clean.drop(columns=[col])
            y = df_clean[col]
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate R-squared
            r_squared = model.score(X, y)
            
            # Calculate VIF: VIF = 1 / (1 - R²)
            if r_squared < 0.9999:  # Avoid division by zero
                vif_value = 1 / (1 - r_squared)
            else:
                vif_value = float('inf')
            
            vif_data.append({'Feature': col, 'VIF': vif_value})
            
            status = "🔴 High Multicollinearity" if vif_value > VIF_THRESHOLD else "✓ OK"
            print(f"{col}: VIF = {vif_value:.2f} {status}")
        except Exception as e:
            print(f"⚠️ Could not calculate VIF for {col}: {str(e)}")
    
    vif_df = pd.DataFrame(vif_data)
    return vif_df


def check_logical_validity(df, numerical_cols):
    """
    Check for logically invalid values (e.g., negative age, impossible values)
    
    Returns:
    --------
    validity_issues : dict
        Dictionary of validity issues found
    """
    print("\n--- Logical Validity Check ---")
    
    validity_issues = {}
    
    for col in numerical_cols:
        issues = []
        
        # Check for negative values in features that should be positive
        if 'age' in col.lower() or 'count' in col.lower() or 'level' in col.lower():
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Negative values: {negative_count}")
        
        # Check for unrealistic age values
        if 'age' in col.lower():
            unrealistic_high = (df[col] > 120).sum()
            unrealistic_low = (df[col] < 0).sum()
            if unrealistic_high > 0:
                issues.append(f"Age > 120: {unrealistic_high}")
            if unrealistic_low > 0:
                issues.append(f"Age < 0: {unrealistic_low}")
        
        if issues:
            validity_issues[col] = issues
            print(f"⚠️ {col}: {', '.join(issues)}")
    
    if not validity_issues:
        print("✓ No logical validity issues found!")
    
    return validity_issues


def identify_rare_categories(df, categorical_cols, threshold=RARE_CATEGORY_THRESHOLD):
    """
    Identify rare categories in categorical features
    
    Returns:
    --------
    rare_categories : dict
        Dictionary of rare categories by feature
    """
    print(f"\n--- Rare Categories (< {threshold}% frequency) ---")
    
    rare_categories = {}
    
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True) * 100
        rare = value_counts[value_counts < threshold]
        
        if len(rare) > 0:
            rare_categories[col] = rare.to_dict()
            print(f"\n{col}:")
            for category, percentage in rare.items():
                print(f"  '{category}': {percentage:.2f}%")
    
    if not rare_categories:
        print("✓ No rare categories found!")
    
    return rare_categories


def perform_groupby_analysis(df, categorical_cols, numerical_cols, target_col):
    """
    Perform GroupBy aggregations for insights
    
    Returns:
    --------
    groupby_results : dict
        Dictionary of aggregation results
    """
    print("\n--- GroupBy Aggregations ---")
    
    groupby_results = {}
    
    # Group by categorical features and aggregate numerical features
    for cat_col in categorical_cols[:3]:  # Limit to first 3 to avoid too much output
        if cat_col != target_col and len(df[cat_col].unique()) < 10:
            print(f"\nGrouping by '{cat_col}':")
            grouped = df.groupby(cat_col)[numerical_cols].mean()
            print(grouped)
            groupby_results[cat_col] = grouped
    
    # If target exists, group by target
    if target_col and target_col in df.columns:
        print(f"\nGrouping by target '{target_col}':")
        grouped_target = df.groupby(target_col)[numerical_cols].mean()
        print(grouped_target)
        groupby_results[target_col] = grouped_target
    
    return groupby_results


def perform_pca(df, numerical_cols, n_components=2):
    """
    Perform Principal Component Analysis
    
    Returns:
    --------
    pca_result : dict
        Dictionary containing PCA results and transformed data
    """
    print(f"\n--- PCA Analysis ({n_components} components) ---")
    
    # Prepare data
    df_pca = df[numerical_cols].dropna()
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pca)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    
    # Create DataFrame
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Print explained variance
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    pca_result = {
        'pca_df': pca_df,
        'explained_variance': pca.explained_variance_ratio_,
        'components': pca.components_,
        'feature_names': numerical_cols
    }
    
    return pca_result


def perform_clustering(df, numerical_cols, n_clusters=3):
    """
    Perform K-Means clustering analysis
    
    Returns:
    --------
    cluster_result : dict
        Dictionary containing cluster labels and centers
    """
    print(f"\n--- K-Means Clustering (k={n_clusters}) ---")
    
    # Prepare data
    df_cluster = df[numerical_cols].dropna()
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    # Print cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster Distribution:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    cluster_result = {
        'labels': cluster_labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }
    
    return cluster_result


def generate_eda_summary(df, numerical_cols, categorical_cols, target_col):
    """
    Generate a comprehensive EDA summary with key findings
    """
    print("\n" + "="*80)
    print("EDA SUMMARY & KEY FINDINGS")
    print("="*80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Total features: {len(df.columns)}")
    print(f"  - Numerical features: {len(numerical_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    
    if target_col:
        print(f"\n🎯 Target Variable: '{target_col}'")
        print(f"  - Classes: {df[target_col].nunique()}")
        print(f"  - Distribution:\n{df[target_col].value_counts()}")
    
    print(f"\n📈 Numerical Features Summary:")
    print(f"  - Mean values range: {df[numerical_cols].mean().min():.2f} to {df[numerical_cols].mean().max():.2f}")
    print(f"  - High correlation pairs: Check correlation heatmap")
    
    print(f"\n📋 Categorical Features Summary:")
    for col in categorical_cols[:5]:  # First 5
        print(f"  - {col}: {df[col].nunique()} unique values")
    
    print(f"\n💡 Key Findings:")
    print(f"  - Review skewness/kurtosis for transformation needs")
    print(f"  - Check VIF values for multicollinearity")
    print(f"  - Examine rare categories for potential grouping")
    print(f"  - Consider PCA for dimensionality reduction")
    print(f"  - Cluster analysis may reveal natural groupings")