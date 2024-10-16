# Data manipulation and analysis
import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt
import csv
import os
from itertools import combinations

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
from scipy import stats

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
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

def jackknife_feature_selection(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 10, n_estimators: int = 100) -> list:
    """
    Perform jackknife feature selection using ANOVA F-value for feature ranking.

    Parameters:
    - X (pd.DataFrame): Feature set, can be multi-indexed
    - y (pd.Series or array-like): Target variable
    - n_features_to_select (int, optional): Number of top features to select. Defaults to 10.
    - n_estimators (int, optional): Not used in this implementation, kept for consistency. Defaults to 100.

    Returns:
    - list: Names of the selected features

    This function implements a jackknife procedure for feature selection. It iteratively removes
    one sample, ranks features using ANOVA F-test, and counts how often each feature is selected.
    The top features are then returned based on their selection frequency.
    """
    # Convert multi-indexed DataFrame to numpy array
    X_train = X.values
    y_train = y.values if isinstance(y, pd.Series) else y
    
    feature_counts = Counter()
    
    # Jackknife procedure
    for i in range(X_train.shape[0]):
        # Remove one sample
        X_train_jackknife = np.delete(X_train, i, axis=0)
        y_train_jackknife = np.delete(y_train, i)
        
        # Rank features using ANOVA F-test
        selector = SelectKBest(f_classif, k=n_features_to_select)
        selector.fit(X_train_jackknife, y_train_jackknife)
        
        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        
        # Update feature counts
        feature_counts.update(selected_features)
    
    # Get top features based on selection frequency
    top_features = [feature for feature, count in feature_counts.most_common(n_features_to_select)]
    
    # Map feature indices back to column names
    feature_names = X.columns.tolist()
    selected_feature_names = [feature_names[i] for i in top_features]
    
    return selected_feature_names

def evaluate_feature_subset(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, 
                            feature_subset: list, n_estimators: int = 100) -> float:
    """
    Evaluate a subset of features using a Random Forest Classifier.

    Parameters:
    - X_train (pd.DataFrame): Training feature set
    - X_test (pd.DataFrame): Test feature set
    - y_train (pd.Series): Training target variable
    - y_test (pd.Series): Test target variable
    - feature_subset (list): List of feature names to evaluate
    - n_estimators (int, optional): Number of trees in the Random Forest. Defaults to 100.

    Returns:
    - float: Accuracy score of the Random Forest Classifier on the test set

    This function trains a Random Forest Classifier on the specified subset of features
    and evaluates its performance on the test set.
    """
    # Select features from multi-indexed DataFrame
    X_subset = X_train[feature_subset]
    X_test_subset = X_test[feature_subset]
    
    # Initialize and train the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_subset, y_train)
    
    # Evaluate the model on the test set
    accuracy = rf.score(X_test_subset, y_test)
    return accuracy

def feature_selection_and_evaluation(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, 
                                     n_features_to_select: int, n_estimators: int) -> pd.DataFrame:
    """
    Perform feature selection using jackknife method and evaluate the selected feature subset.

    Parameters:
    - X_train (pd.DataFrame): Training feature set
    - X_val (pd.DataFrame): Validation feature set
    - y_train (pd.Series): Training target variable
    - y_val (pd.Series): Validation target variable
    - n_features_to_select (int): Number of features to select
    - n_estimators (int): Number of estimators for the Random Forest classifier

    Returns:
    - pd.DataFrame: A DataFrame summarizing the results of feature selection and evaluation

    This function performs jackknife feature selection, evaluates the selected feature subset,
    compares it with using all features, and returns a summary of the results.
    """
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
        'Number of Estimators': [n_estimators],
        'Number of Selected Features': [n_features_to_select],
        'Accuracy with Selected Features': [accuracy_selected],
        'Accuracy with All Features': [accuracy_all],
        'Selected Features': [selected_features]
    })

    return result_df

def perform_anova(df: pd.DataFrame, feature_columns: list, pd_column: str, gender_column: str) -> pd.DataFrame:
    """
    Perform one-way ANOVA for each feature, separated by gender, to compare Parkinson's and non-Parkinson's groups.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data
    - feature_columns (list): List of feature column names to analyze
    - pd_column (str): Name of the column indicating Parkinson's Disease status (1 for PD, 0 for non-PD)
    - gender_column (str): Name of the column indicating gender (0 for Female, 1 for Male)

    Returns:
    - pd.DataFrame: A DataFrame containing ANOVA results with columns 'Feature', 'Gender', 'F-value', and 'p-value'

    This function performs a one-way ANOVA for each feature in feature_columns, separately for each gender,
    comparing the Parkinson's Disease group with the non-Parkinson's Disease group.
    """
    results = []
    for feature in feature_columns:
        for gender in df[gender_column].unique():
            # Subset data for the current gender
            feature_data = df[df[gender_column] == gender]
            
            # Separate PD and non-PD groups
            pd_group = feature_data[feature_data[pd_column] == 1][feature]
            non_pd_group = feature_data[feature_data[pd_column] == 0][feature]
            
            # Perform one-way ANOVA
            f_value, p_value = stats.f_oneway(pd_group, non_pd_group)
            
            # Convert gender code to string label
            gender_label = 'Female' if gender == 0 else 'Male'
            
            # Store results
            results.append({
                'Feature': feature,
                'Gender': gender_label,
                'F-value': f_value,
                'p-value': p_value
            })
    
    # Convert results to DataFrame and return
    return pd.DataFrame(results)

def plot_feature_distributions(df: pd.DataFrame, feature_columns: list, pd_column: str, gender_column: str, anova_results: pd.DataFrame) -> None:
    """
    Plot feature distributions by Parkinson's Disease status and gender using violin plots.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data
    - feature_columns (list): List of feature column names to plot
    - pd_column (str): Name of the column indicating Parkinson's Disease status (0 for Healthy, 1 for Parkinson's)
    - gender_column (str): Name of the column indicating gender (0 for Female, 1 for Male)
    - anova_results (pd.DataFrame): DataFrame containing ANOVA results with 'Feature' and 'p-value' columns

    Returns:
    - None: Displays the plot grid
    """
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = (n_features - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    # Create mapping functions for gender and PD status
    gender_map = lambda x: 'Female' if x == 0 else 'Male'
    pd_map = lambda x: 'Healthy' if x == 0 else 'Parkinson'

    for i, feature in enumerate(feature_columns):
        ax = axes[i // n_cols, i % n_cols]
        
        # Create a copy of the dataframe with mapped values
        plot_df = df.copy()
        plot_df[gender_column] = plot_df[gender_column].map(gender_map)
        plot_df[pd_column] = plot_df[pd_column].map(pd_map)
        
        # Create violin plot
        sns.violinplot(x=gender_column, y=feature, hue=pd_column, 
                       data=plot_df, split=True, inner="box", ax=ax)
        
        # Get p-value from ANOVA results
        p_value = anova_results.loc[anova_results['Feature'] == str(feature), 'p-value'].values[0]

        # Set plot title and labels
        ax.set_title(feature)
        ax.set_xlabel('')
        if i % n_cols != 0:
            ax.set_ylabel('')
        
        # Adjust legend
        if i % 3 == 0:  # Only show legend for the first plot in each row
            ax.legend(title='Status')
        else:
            ax.get_legend().remove()

    # Remove any unused subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    # Set overall title
    fig.suptitle('Feature Distributions by PD Status and Gender', fontsize=16, y=1.0)

    plt.tight_layout()
    plt.show()


def fill_empty_with_previous(lst: list[str]) -> list[str]:
    """Fills empty strings in the list with the last non-empty value.

    Args:
        lst (list[str]): A list of strings where some elements may be empty strings ('').

    Returns:
        list[str]: The modified list where empty strings are replaced by the last non-empty value.
    
    Example:
        fill_empty_with_previous(['','a', '', 'b', '', '']) -> ['','a', 'a', 'b', 'b', 'b']
    """
    last_value = ''  # Initialize variable to store the last non-empty string encountered

    for index, item in enumerate(lst):
        # Skip processing for the last item in the list
        # This is specific to the pd_speech_features dataset to avoid labelling the class
        if index == len(lst) - 1:
            continue

        # If the current item is not empty, update last_value
        elif item != '':
            last_value = item
        
        # If the current item is empty, replace it with last_value
        else:
            lst[index] = last_value

    return lst

def plot_missing_values_by_class(df: pd.DataFrame, class_column: str, labels: list = []) -> None:
    """
    Plots a heatmap of the percentage of missing values for each feature, grouped by a specified class column.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data
    - class_column (str): The name of the column in `df` to group by (i.e., the class)
    - labels (list, optional): A list of labels for the y-axis (class labels). Defaults to an empty list

    Returns:
    - None: The function generates and displays a heatmap plot

    Raises:
    - ValueError: If the class_column does not exist in the dataframe

    Example:
        plot_missing_values_by_class(df, 'Gender', ['Female', 'Male'])
    """
    # Check if class_column exists in the DataFrame
    if class_column not in df.columns:
        raise ValueError(f"'{class_column}' does not exist in the dataframe.")

    # Calculate percentage of missing values for each feature and class
    missing_percentages = df.groupby(class_column).apply(lambda x: x.isnull().mean())
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))  # Set figure size
    heatmap = sns.heatmap(missing_percentages, 
                          cmap='YlOrRd',  # Color map for the heatmap
                          cbar_kws={'label': 'Percentage of Missing Values', 
                                    'format': '%.2f'})  # Format for color bar ticks
    
    plt.title(f'Missing Values by {class_column}')  # Set title for the figure
    
    # Add custom y-axis labels if provided
    if labels:
        heatmap.set_yticklabels(labels)

    # Customize the color bar
    colorbar = heatmap.collections[0].colorbar
    max_percentage = missing_percentages.values.max()
    colorbar.set_ticks([0, max_percentage, 1])  # Set tick positions on the color bar
    colorbar.set_ticklabels(['0%', f'{max_percentage*100:.0f}%', '100%'])  # Set tick labels

    # Set axis labels
    heatmap.set(xlabel="Feature", ylabel=class_column)

    plt.show()  # Display the plot


def analyze_missing_values_by_class(df: pd.DataFrame, class_column: str) -> None:
    """
    Analyzes missing values in a DataFrame by class and performs statistical tests.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data
    - class_column (str): Name of the column to group by (i.e., the class)

    Returns:
    - None: Prints disparity in missing values and significant relationships

    Raises:
    - ValueError: If class_column does not exist in the dataframe

    Example:
        analyze_missing_values_by_class(df, 'Gender')
    """
    # Check if class_column exists in the DataFrame
    if class_column not in df.columns:
        raise ValueError(f"'{class_column}' does not exist in the dataframe.")
    
    # Calculate percentage of missing values for each feature and class
    missing_by_class = df.groupby(class_column).apply(lambda x: x.isnull().mean())
    
    # Calculate the difference in missing value percentages between classes
    class_disparity = missing_by_class.diff().iloc[-1].abs().sort_values(ascending=False)
    
    print(f"Disparity in missing values between {class_column} classes:")
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


def plot_missing_values_distribution(df: pd.DataFrame, axes: int = 0) -> None:
    """
    Plots the distribution of missing values for rows or columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data
    - axes (int): Axis along which to calculate missing values. 
                  0 for columns, 1 for rows. Defaults to 0 (columns)

    Returns:
    - None: Displays the histogram plot

    Example:
        # Plot missing values distribution per column
        plot_missing_values_distribution(df, axes=0)
        
        # Plot missing values distribution per row
        plot_missing_values_distribution(df, axes=1)
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate missing values
    if axes == 1:
        missing_counts = df.isnull().sum(axis=1)
        title = 'Distribution of Missing Values per Row'
    elif axes == 0:
        missing_counts = df.isnull().sum()
        title = 'Distribution of Missing Values per Column'
    else:
        raise ValueError("axes must be 0 for columns or 1 for rows")

    # Plot histogram
    plt.hist(missing_counts, bins=20)
    
    # Set plot attributes
    plt.xlim(0)
    plt.title(title)
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Frequency')
    
    # Display the plot
    plt.show()
                
def fill_missing_values(df: pd.DataFrame, columns: list[str], method: str = 'mean') -> pd.DataFrame:
    """
    Fills missing values in one or more specified columns of a DataFrame using the mean, median, or mode.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - columns (list[str]): List of column names in which to fill missing values
    - method (str): The method to fill missing values: 'mean', 'median', or 'mode'. Default is 'mean'
    
    Returns:
    - pd.DataFrame: A DataFrame with the missing values filled in the specified column(s)
    
    Raises:
    - ValueError: If a specified column doesn't exist, if the method is invalid, or if mode can't be calculated
    """
    # Ensure that 'columns' is a list even if a single column is provided
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        # Check if the column exists in the DataFrame
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

def compute_proximity_matrix(forest: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Compute the proximity matrix for an Isolation Forest.

    This function calculates a proximity matrix based on the Isolation Forest algorithm.
    The proximity between two data points is defined as the fraction of trees in the forest
    where these points end up in the same leaf node.

    Parameters:
    - forest (IsolationForest): A fitted Isolation Forest model
    - X (np.ndarray): The input samples, shape (n_samples, n_features)

    Returns:
    - np.ndarray: The proximity matrix, shape (n_samples, n_samples)
    """
    # Get the number of samples
    n_samples = X.shape[0]
    
    # Initialize the proximity matrix with zeros
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
                    proximity_matrix[j, i] += 1  # Ensure symmetry of the matrix
    
    # Normalize the proximity matrix by the number of trees
    return proximity_matrix / len(forest.estimators_)

def check_binary_column(column: pd.Series, column_name: str) -> None:
    """
    Check if a column contains only binary values (0 and 1).
    
    Parameters:
    - column (pd.Series): The column to check
    - column_name (str): The name of the column being checked
    
    Returns:
    - None
    
    Raises:
    - ValueError: If the column contains values other than 0 and 1
    """
    # Get unique values in the column
    unique_values = column.unique()
    
    # Check if all unique values are either 0 or 1
    if set(unique_values).issubset({0, 1}):
        print(f"{column_name} contains only 0 and 1.")
    else:
        # Raise an error if other values are found
        raise ValueError(f"Error: {column_name} contains values other than 0 and 1: {unique_values}")

def correct_outliers(df, columns, outlier_column, method='mean'):
    """
    Correct outliers in specified columns based on a boolean outlier indicator column.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - columns (list): List of column names to correct
    - outlier_column (str): Name of the boolean column indicating outliers (True for outliers)
    - method (str): Method to use for correction ('mean', 'median', 'mode', or 'winsorize')
    
    Returns:
    - pd.DataFrame: DataFrame with corrected outliers
    """
    df_corrected = df.copy()
    
    for column in columns:
        if method == 'mean':
            # Replace outliers with mean of non-outlier values
            mean_value = df_corrected.loc[~df_corrected[outlier_column], column].mean()
            df_corrected.loc[df_corrected[outlier_column], column] = mean_value
        
        elif method == 'median':
            # Replace outliers with median of non-outlier values
            median_value = df_corrected.loc[~df_corrected[outlier_column], column].median()
            df_corrected.loc[df_corrected[outlier_column], column] = median_value
        
        elif method == 'mode':
            # Replace outliers with mode of non-outlier values
            mode_value = df_corrected.loc[~df_corrected[outlier_column], column].mode().iloc[0]
            df_corrected.loc[df_corrected[outlier_column], column] = mode_value
        
        elif method == 'winsorize':
            # Winsorize outliers to the nearest non-outlier value
            sorted_values = df_corrected.loc[~df_corrected[outlier_column], column].sort_values()
            lower_bound = sorted_values.iloc[5]
            upper_bound = sorted_values.iloc[-5]
            df_corrected.loc[df_corrected[outlier_column], column] = df_corrected.loc[df_corrected[outlier_column], column].clip(lower_bound, upper_bound)
        
        else:
            raise ValueError("Invalid method. Choose 'mean', 'median', 'mode', or 'winsorize'.")
    
    return df_corrected
