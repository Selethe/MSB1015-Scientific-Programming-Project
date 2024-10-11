# Data manipulation and analysis
import pandas as pd
import numpy as np
import csv
from collections import Counter
from math import sqrt
import os
from itertools import combinations

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Progress bar for loops
from tqdm.notebook import tqdm

# Additional NumPy function
from numpy import interp

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def jackknife_feature_selection(X, y, n_features_to_select=10, n_estimators=100):
    # Convert multi-indexed DataFrame to numpy array
    X_train = X.values
    y_train = y.values if isinstance(y, pd.Series) else y
    
    feature_counts = Counter()
    
    # Jackknife procedure
    for i in range(X_train.shape[0]):
        # Remove one sample
        X_train_jackknife = np.delete(X_train, i, axis=0)
        y_train_jackknife = np.delete(y_train, i)
        
        # Rank features
        selector = SelectKBest(f_classif, k=n_features_to_select)
        selector.fit(X_train_jackknife, y_train_jackknife)
        
        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        
        # Update feature counts
        feature_counts.update(selected_features)
    
    # Get top features
    top_features = [feature for feature, count in feature_counts.most_common(n_features_to_select)]
    
    # Map feature indices back to column names
    feature_names = X.columns.tolist()
    selected_feature_names = [feature_names[i] for i in top_features]
    
    return selected_feature_names

def evaluate_feature_subset(X_train, X_test, y_train, y_test, feature_subset, n_estimators=100):
    # Select features from multi-indexed DataFrame
    X_subset = X_train[feature_subset]
    X_test_subset = X_test[feature_subset]
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_subset, y_train)
    
    accuracy = rf.score(X_test_subset, y_test)
    return accuracy

def feature_selection_and_evaluation(X_train, X_val, y_train, y_val, n_features_to_select, n_estimators):
    # Perform jackknife feature selection
    selected_features = jackknife_feature_selection(X_train, y_train, n_features_to_select=n_features_to_select, n_estimators=n_estimators)
    print("Selected features:", selected_features)

    # Evaluate the selected feature subset
    accuracy_selected = evaluate_feature_subset(X_train, X_val, y_train, y_val, selected_features, n_estimators=n_estimators)
    print(f"Accuracy with selected features: {accuracy_selected:.4f}")

    # Compare with using all features
    accuracy_all = evaluate_feature_subset(X_train, X_val, y_train, y_val, X_train.columns, n_estimators=n_estimators)
    print(f"Accuracy with all features: {accuracy_all:.4f}")

    # Create a summary DataFrame
    result_df = pd.DataFrame({
        'Number of Estimators': n_estimators,
        'Number of Selected Features': n_features_to_select,
        'Accuracy with Selected Features': accuracy_selected,
        'Accuracy with All Features': accuracy_all,
        'Selected Features': [selected_features]
    })

    return result_df

def perform_anova(df, feature_columns, pd_column, gender_column):
    results = []
    for feature in feature_columns:
        for gender in df[gender_column].unique():
            feature_data = df[df[gender_column] == gender]
            
            pd_group = feature_data[feature_data[pd_column] == 1][feature]
            non_pd_group = feature_data[feature_data[pd_column] == 0][feature]
            
            f_value, p_value = stats.f_oneway(pd_group, non_pd_group)
            if gender == 0:
                gender = 'Female'
            else:
                gender = 'Male'
            results.append({
                'Feature': feature,
                'Gender': gender,
                'F-value': f_value,
                'p-value': p_value
            })
    
    return pd.DataFrame(results)

def plot_feature_distributions(df, feature_columns, pd_column, gender_column, anova_results):
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = (n_features - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    # Create mapping functions for gender and PD status
    gender_map = lambda x: 'Female' if x == 0 else 'Male'
    pd_map = lambda x: 'Healthy' if x == 0 else 'Parkinson'

    def add_stars(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    for i, feature in enumerate(feature_columns):
        ax = axes[i // n_cols, i % n_cols]
        
        # Create a copy of the dataframe with mapped values
        plot_df = df.copy()
        plot_df[gender_column] = plot_df[gender_column].map(gender_map)
        plot_df[pd_column] = plot_df[pd_column].map(pd_map)
        
        sns.violinplot(x=gender_column, y=feature, hue=pd_column, 
                       data=plot_df, split=True, inner="box", ax=ax)
        
        # Add stars if p-value is significant
        p_value = anova_results.loc[anova_results['Feature'] == str(feature), 'p-value'].values[0]
        stars = add_stars(p_value)
        
        if stars:
            y_max = plot_df[feature].max()
            y_range = plot_df[feature].max() - plot_df[feature].min()
            y_pos = y_max + 0.15 * y_range
            
            ax.text(0, y_pos, stars, ha='center', va='bottom')
            ax.text(1, y_pos, stars, ha='center', va='bottom')
        
        ax.set_title(feature)
        ax.set_xlabel('')
        if i % n_cols != 0:
            ax.set_ylabel('')
        
        # Adjust legend
        if i % 3 == 0:  # Only show legend for the first plot
            ax.legend(title='Status')
        else:
            ax.get_legend().remove()

    # Remove any unused subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    fig.suptitle('Feature Distributions by PD Status and Gender', fontsize=16, y=1.0)

    plt.tight_layout()
    plt.show()
