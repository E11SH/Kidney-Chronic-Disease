"""
Visualization Module
Contains all plotting and visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from config import DPI, OUTPUT_DIR


# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_numerical_distributions(df, numerical_cols, save_path='week1_numerical_distributions.png'):
    """Create histograms for all numerical features"""
    if len(numerical_cols) == 0:
        print("⚠️ No numerical columns to plot")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Numerical distributions saved as '{save_path}'")
    plt.close()


def plot_boxplots(df, numerical_cols, save_path='week1_boxplots.png'):
    """Create box plots for all numerical features"""
    if len(numerical_cols) == 0:
        print("⚠️ No numerical columns to plot")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        bp = axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[idx].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Box plots saved as '{save_path}'")
    plt.close()


def plot_categorical_distributions(df, categorical_cols, save_path='week1_categorical_distributions.png'):
    """Create bar plots for all categorical features"""
    if len(categorical_cols) == 0:
        print("⚠️ No categorical columns to plot")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        colors = plt.cm.Set3(range(len(value_counts)))
        axes[idx].bar(range(len(value_counts)), value_counts.values, color=colors, 
                     edgecolor='black', linewidth=1.5)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].grid(axis='y', alpha=0.3)
    
    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Categorical distributions saved as '{save_path}'")
    plt.close()


def plot_correlation_heatmap(df, numerical_cols, save_path='correlation_heatmap.png'):
    """Create correlation heatmap for numerical features"""
    if len(numerical_cols) < 2:
        print("⚠️ Need at least 2 numerical columns for correlation")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved as '{save_path}'")
    plt.close()


def plot_missing_values(df, save_path='missing_values.png'):
    """Visualize missing values"""
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    missing_values = df.isnull().sum()
    missing_summary = missing_values[missing_values > 0].sort_values(ascending=False)
    total_missing = missing_values.sum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if total_missing > 0:
        missing_summary.plot(kind='bar', color='coral', edgecolor='black', linewidth=1.5)
        plt.title('Missing Values by Column', fontsize=16, fontweight='bold')
        plt.xlabel('Columns', fontsize=12)
        plt.ylabel('Number of Missing Values', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(missing_summary.values):
            plt.text(i, v + max(missing_summary.values)*0.02, str(v), 
                    ha='center', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Missing Values Found! ✓', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=24, fontweight='bold', color='green',
                transform=ax.transAxes)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Missing values visualization saved as '{save_path}'")
    plt.close()


def plot_duplicates(df, save_path='duplicate_rows.png'):
    """Visualize duplicate rows"""
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    duplicates_count = df.duplicated().sum()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    duplicate_status = ['Unique Rows', 'Duplicate Rows']
    duplicate_counts = [len(df) - duplicates_count, duplicates_count]
    colors = ['#2ecc71', '#e74c3c']
    
    plt.bar(duplicate_status, duplicate_counts, color=colors, edgecolor='black', linewidth=2)
    plt.title('Duplicate Rows Analysis', fontsize=16, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    
    for i, v in enumerate(duplicate_counts):
        plt.text(i, v + max(duplicate_counts)*0.02, str(v), 
                ha='center', fontweight='bold', fontsize=14)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Duplicate rows visualization saved as '{save_path}'")
    plt.close()


def plot_outliers(df, numerical_cols, save_path='outlier_detection.png'):
    """Detect and visualize outliers using IQR method"""
    if len(numerical_cols) == 0:
        print("⚠️ No numerical columns for outlier detection")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    outlier_info = {}
    
    for idx, col in enumerate(numerical_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = len(outliers)
        
        bp = axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[idx].axhline(y=lower_bound, color='r', linestyle='--', linewidth=2, label='Lower Bound')
        axes[idx].axhline(y=upper_bound, color='r', linestyle='--', linewidth=2, label='Upper Bound')
        axes[idx].set_title(f'Outliers in {col}\n({len(outliers)} outliers)', 
                           fontweight='bold', fontsize=11)
        axes[idx].set_ylabel(col)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(axis='y', alpha=0.3)
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Outlier detection visualization saved as '{save_path}'")
    plt.close()
    
    return outlier_info


def plot_scatter_matrix(df, numerical_cols, save_path='scatter_numerical.png'):
    """Create scatter plots between numerical features"""
    if len(numerical_cols) < 2:
        print("⚠️ Need at least 2 numerical columns for scatter plots")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # Select up to 5 features to avoid overcrowding
    selected_cols = numerical_cols[:5]
    n_features = len(selected_cols)
    
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                axes[i, j].hist(df[selected_cols[i]].dropna(), bins=20, color='skyblue', edgecolor='black')
            else:
                axes[i, j].scatter(df[selected_cols[j]], df[selected_cols[i]], alpha=0.5, s=10)
            
            if i == n_features - 1:
                axes[i, j].set_xlabel(selected_cols[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(selected_cols[i], fontsize=10)
    
    plt.suptitle('Scatter Matrix of Numerical Features', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Scatter matrix saved as '{save_path}'")
    plt.close()


def plot_categorical_vs_numerical(df, categorical_cols, numerical_cols, save_path='categorical_vs_numerical.png'):
    """Create box plots showing categorical vs numerical relationships"""
    if len(categorical_cols) == 0 or len(numerical_cols) == 0:
        print("⚠️ Need both categorical and numerical columns")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # Select first categorical and first 3 numerical features
    cat_col = categorical_cols[0]
    num_cols_selected = numerical_cols[:3]
    
    fig, axes = plt.subplots(1, len(num_cols_selected), figsize=(18, 5))
    if len(num_cols_selected) == 1:
        axes = [axes]
    
    for idx, num_col in enumerate(num_cols_selected):
        df_plot = df[[cat_col, num_col]].dropna()
        df_plot.boxplot(column=num_col, by=cat_col, ax=axes[idx])
        axes[idx].set_title(f'{num_col} by {cat_col}')
        axes[idx].set_xlabel(cat_col)
        axes[idx].set_ylabel(num_col)
        plt.sca(axes[idx])
        plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Categorical vs Numerical plots saved as '{save_path}'")
    plt.close()


def plot_categorical_vs_target(df, categorical_cols, target_col, save_path='categorical_vs_target.png'):
    """Create bar plots showing categorical features vs target"""
    if not target_col or len(categorical_cols) == 0:
        print("⚠️ Need target column and categorical columns")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # Select first 3 categorical features (excluding target)
    cat_cols_selected = [col for col in categorical_cols if col != target_col][:3]
    
    if len(cat_cols_selected) == 0:
        print("⚠️ No categorical columns available (excluding target)")
        return
    
    fig, axes = plt.subplots(1, len(cat_cols_selected), figsize=(18, 5))
    if len(cat_cols_selected) == 1:
        axes = [axes]
    
    for idx, cat_col in enumerate(cat_cols_selected):
        cross_tab = pd.crosstab(df[cat_col], df[target_col], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=False, ax=axes[idx], colormap='Set2')
        axes[idx].set_title(f'{cat_col} vs {target_col}', fontweight='bold')
        axes[idx].set_xlabel(cat_col)
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].legend(title=target_col)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Categorical vs Target plots saved as '{save_path}'")
    plt.close()


def plot_pairplot(df, numerical_cols, target_col=None, save_path='pairplot.png'):
    """Create pairplot for multiple numerical features"""
    if len(numerical_cols) < 2:
        print("⚠️ Need at least 2 numerical columns for pairplot")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # Select up to 4 features to keep plot readable
    selected_cols = numerical_cols[:4]
    
    if target_col and target_col in df.columns:
        plot_df = df[selected_cols + [target_col]].dropna()
        sns.pairplot(plot_df, hue=target_col, diag_kind='kde', plot_kws={'alpha': 0.6})
    else:
        plot_df = df[selected_cols].dropna()
        sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
    
    plt.suptitle('Pairplot of Numerical Features', y=1.01, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Pairplot saved as '{save_path}'")
    plt.close()


def plot_3d_scatter(df, numerical_cols, target_col=None, save_path='3d_scatter.png'):
    """Create 3D scatter plot"""
    if len(numerical_cols) < 3:
        print("⚠️ Need at least 3 numerical columns for 3D scatter")
        return
    
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    # Select first 3 numerical features
    x_col, y_col, z_col = numerical_cols[:3]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    df_plot = df[[x_col, y_col, z_col]].dropna()
    
    if target_col and target_col in df.columns:
        # Color by target
        for target_val in df[target_col].unique():
            mask = df[target_col] == target_val
            df_subset = df[mask][[x_col, y_col, z_col]].dropna()
            ax.scatter(df_subset[x_col], df_subset[y_col], df_subset[z_col], 
                      label=str(target_val), alpha=0.6, s=30)
        ax.legend()
    else:
        ax.scatter(df_plot[x_col], df_plot[y_col], df_plot[z_col], 
                  c='skyblue', alpha=0.6, s=30)
    
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_zlabel(z_col, fontsize=10)
    ax.set_title('3D Scatter Plot', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ 3D scatter plot saved as '{save_path}'")
    plt.close()


def plot_pca_results(pca_result, target_col=None, df=None, save_path='pca_analysis.png'):
    """Visualize PCA results"""
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    pca_df = pca_result['pca_df']
    explained_variance = pca_result['explained_variance']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot of first 2 components
    if target_col and df is not None and target_col in df.columns:
        target_aligned = df[target_col].iloc[pca_df.index]
        for target_val in target_aligned.unique():
            mask = target_aligned == target_val
            axes[0].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                          label=str(target_val), alpha=0.6, s=30)
        axes[0].legend()
    else:
        axes[0].scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=30, c='skyblue')
    
    axes[0].set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=12)
    axes[0].set_title('PCA: First Two Components', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Explained variance plot
    components = range(1, len(explained_variance) + 1)
    cumulative_variance = np.cumsum(explained_variance)
    
    axes[1].bar(components, explained_variance, alpha=0.7, color='skyblue', label='Individual')
    axes[1].plot(components, cumulative_variance, 'ro-', linewidth=2, label='Cumulative')
    axes[1].set_xlabel('Principal Component', fontsize=12)
    axes[1].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[1].set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ PCA visualization saved as '{save_path}'")
    plt.close()


def plot_clusters(cluster_result, pca_result=None, save_path='cluster_analysis.png'):
    """Visualize clustering results"""
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    labels = cluster_result['labels']
    
    if pca_result is not None:
        pca_df = pca_result['pca_df']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=labels, 
                           cmap='viridis', alpha=0.6, s=50)
        
        # Plot cluster centers if in 2D PCA space
        if 'centers' in cluster_result:
            centers_2d = cluster_result['centers'][:, :2]  # First 2 PCA components
            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                      c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                      label='Centroids')
        
        plt.colorbar(scatter, label='Cluster')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('K-Means Clustering (PCA Space)', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Cluster visualization saved as '{save_path}'")
        plt.close()
    else:
        print("⚠️ PCA results needed for cluster visualization")


def plot_skewness_kurtosis(skew_kurt_df, save_path='skewness_kurtosis.png'):
    """Visualize skewness and kurtosis"""
    # Add OUTPUT_DIR to save path
    save_path = os.path.join(OUTPUT_DIR, save_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Skewness
    axes[0].barh(skew_kurt_df['Feature'], skew_kurt_df['Skewness'], color='coral', edgecolor='black')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].axvline(x=1, color='red', linestyle='--', linewidth=1, label='Threshold')
    axes[0].axvline(x=-1, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Skewness', fontsize=12)
    axes[0].set_title('Skewness of Features', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Kurtosis
    axes[1].barh(skew_kurt_df['Feature'], skew_kurt_df['Kurtosis'], color='skyblue', edgecolor='black')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Kurtosis', fontsize=12)
    axes[1].set_title('Kurtosis of Features', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Skewness/Kurtosis visualization saved as '{save_path}'")
    plt.close()