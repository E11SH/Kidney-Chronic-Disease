"""
Configuration file for Kidney Disease Classification Project
Contains paths, settings, and constants used across modules
"""

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# FILE PATHS
# =============================================================================
RAW_DATA_PATH = "College/kidney.csv"
CLEANED_DATA_PATH = "College/kidney_cleaned.csv"
OUTPUT_DIR = "College/diagrams/"
MODELS_DIR = "College/trained_models/"
RESULTS_DIR = "College/model_results/"

# =============================================================================
# PLOT SETTINGS
# =============================================================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
DPI = 300

# =============================================================================
# DATA PROCESSING SETTINGS
# =============================================================================
# Characters to replace with NaN during cleaning
MISSING_VALUE_INDICATORS = ['?', '\t?', '']

# IQR multiplier for outlier detection
IQR_MULTIPLIER = 1.5

# Random state for reproducibility
RANDOM_STATE = 42

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================
# Threshold for considering a feature as highly skewed
SKEWNESS_THRESHOLD = 1.0

# VIF threshold for multicollinearity
VIF_THRESHOLD = 10.0

# Minimum frequency for rare category identification (%)
RARE_CATEGORY_THRESHOLD = 1.0