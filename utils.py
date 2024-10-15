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
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# Data manipulation and analysis
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest


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


def fill_empty_with_previous(lst: list[str]) -> list[str]:
    """Fills empty strings in the list with the last non-empty value.

    Args: lst (list[str]): A list of strings where some elements may be empty strings ('').

    Returns: list[str]: The modified list where empty strings are replaced by the last non-empty value.
    
    Example:
        fill_empty_with_previous(['','a', '', 'b', '', '']) -> ['','a', 'a', 'b', 'b', 'b']
    """
    last_value = ''  # Stores the last non-empty string encountered

    for index, item in enumerate(lst):
        # Skip processing for the last item in the list, for the pd_speech_features dataset this is to avoid labelling class
        if index == len(lst) - 1:
            continue

        # Update last_value if the current item is not empty
        elif item != '':
            last_value = item
        else:
            # Replace empty string with the last non-empty value
            lst[index] = last_value

    return lst

def plot_missing_values_by_class(df: pd.DataFrame, class_column: str, labels: list = []) -> None:
    """
    Plots a heatmap of the percentage of missing values for each feature, grouped by a specified class column.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        class_column (str): The name of the column in `df` to group by (i.e., the class).
        labels (list, optional): A list of labels for the y-axis (class labels). Defaults to an empty list.

    Returns:
        None: The function generates and displays a heatmap plot.
    
    Raises:
        ValueError: If the class_column does not exist in the dataframe.
    
    Example:
        plot_missing_values_by_class(df, 'Gender', ['female', 'Male',])
    """
    # Calculate percentage of missing values for each feature and class
    missing_percentages = df.groupby(class_column).apply(lambda x: x.isnull().mean())
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))  # Set figure size
    heatmap = sns.heatmap(missing_percentages, 
                           cmap='YlOrRd',  # Color map for the heatmap
                           cbar_kws={'label': 'Percentage of Missing Values', 
                                      'ticks': [0, 100, 4],  # Set ticks on the color bar
                                      'format': '%.2f'})  # Format for color bar ticks
    plt.title('Missing Values by Class')  # Set title for the figure
    
    # Add custom y-axis labels if provided
    if labels:
        heatmap.set_yticklabels(labels)

    # Customize the color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([0, missing_percentages.values.max(), 1])  # Set tick positions on the color bar
    colorbar.set_ticklabels(['0%', f'{missing_percentages.values.max()*100:.0f}%', '100%'])  # Set tick labels
    heatmap.set(title=f"Missing values per {class_column[1]}", xlabel="Feature", ylabel=f"{class_column[1]}")  # Set axis labels
    plt.show()  # Display the plot

    
def analyze_missing_values_by_class(df: pd.DataFrame, class_column: set) -> None:
    """
    Analyzes missing values in a DataFrame by class and performs statistical tests.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        class_column (str): Name of the column to group by (i.e., the class).

    Returns:
        None: Prints disparity in missing values and significant relationships.

    Raises:
        ValueError: If class_column does not exist in the dataframe.

    Expected types:
        df (pd.DataFrame): pandas DataFrame
        class_column (str): string representing the class column name

    Example:
        analyze_missing_values_by_class(df, 'Gender')
    """
    
    # Calculate percentage of missing values for each feature and class
    missing_by_class = df.groupby(class_column).apply(lambda x: x.isnull().mean())
    
    # Calculate the difference in missing value percentages between classes
    class_disparity = missing_by_class.diff().iloc[-1].abs().sort_values(ascending=False)
    
    print(f"Disparity in missing values between {class_column[1]}:")
    display(class_disparity)
    
    # Perform chi-square test for independence
    for column in df.columns:
        if column != class_column:
            contingency_table = pd.crosstab(df[class_column], df[column].isnull())
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            if p_value < 0.05:
                print(f"\nSignificant relationship found for {column}")
                print(f"Chi-square statistic: {chi2:.2f}")
                print(f"p-value: {p_value:.4f}")

                
def plot_missing_values_distribution(df: pd.DataFrame, axes: int=0) -> None:
    """
    Plots the distribution of missing values for rows or columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        axes (int): Axis along which to calculate missing values. 
                                         1 for rows, 0 for columns. Defaults to 0 (columns).

    Returns:
        None: Displays the histogram plot.

    Example:
        # Plot missing values distribution per row
        plot_missing_values_distribution(df, axes=0)
        
        # Plot missing values distribution per column
        plot_missing_values_distribution(df, axes='columns')
    """
    plt.figure(figsize=(10, 6))
    if axes:
        plt.hist(df.isnull().sum(axis=1), bins=20)
    else:
        plt.hist(df.isnull().sum(), bins=20)
    
    plt.xlim(0)
    if axes:
        plt.title(f'Distribution of Missing Values per Rows')
    else:
        plt.title(f'Distribution of Missing Values per Column')
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Frequency')
    plt.show()
                
def fill_missing_values(df: pd.DataFrame, columns: list[str], method:str='mean')->  pd.DataFrame:
    """
    Fills missing values in one or more specified columns of a DataFrame using the mean, median, or mode.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (str or list): The column name or list of column names in which to fill missing values.
    - method (str): The method to fill missing values: 'mean', 'median', or 'mode'. Default is 'mean'.
    
    Returns:
    - pd.DataFrame: A DataFrame with the missing values filled in the specified column(s).
    """
    # Ensure that 'columns' is a list even if a single column is provided
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        
        # Determine the fill value based on the chosen method
        if method == 'mean':
            fill_value = df[column].mean()
        elif method == 'median':
            fill_value = df[column].median()
        elif method == 'mode':
            mode_series = df[column].mode()
            if not mode_series.empty:
                fill_value = mode_series[0]  # Choose the first mode if multiple modes exist
            else:
                raise ValueError(f"Column '{column}' has no mode (all values are missing or unique).")
        else:
            raise ValueError(f"Invalid method '{method}'. Choose 'mean', 'median', or 'mode'.")
        
        # Fill the missing values in the current column with the calculated value
        df[column].fillna(fill_value, inplace=True)
    
    return df

def compute_proximity_matrix(forest:IsolationForest, X:np.ndarray) -> np.ndarray:
    """
    Compute the proximity matrix for an Isolation Forest.

    This function calculates a proximity matrix based on the Isolation Forest algorithm.
    The proximity between two data points is defined as the fraction of trees in the forest
    where these points end up in the same leaf node.

    Parameters:
    forest (IsolationForest): A fitted Isolation Forest model.
    X (array-like): The input samples, shape (n_samples, n_features).

    Returns:
    numpy.ndarray: The proximity matrix, shape (n_samples, n_samples).
    """
    n_samples = X.shape[0]
    proximity_matrix = np.zeros((n_samples, n_samples))
    
    # Iterate through each tree in the forest
    for tree in forest.estimators_:
        # Get the leaf node indices for all samples
        leaves_index = tree.apply(X)
        
        # Compare leaf indices for each pair of samples
        for i in range(n_samples):
            for j in range(i, n_samples):
                # If two samples end up in the same leaf, increment their proximity
                if leaves_index[i] == leaves_index[j]:
                    proximity_matrix[i, j] += 1
                    proximity_matrix[j, i] += 1
    
    # Normalize the proximity matrix by the number of trees
    return proximity_matrix / len(forest.estimators_)

def check_binary_column(column, column_name):
    unique_values = column.unique()
    if set(unique_values).issubset({0, 1}):
        print(f"{column_name} contains only 0 and 1.")
    else:
        raise ValueError(f"Error: {column_name} contains values other than 0 and 1: {unique_values}")