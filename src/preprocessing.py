from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def fill_null_values(dataframe, columns, value):
    """
    Fill null values in specified columns of a DataFrame with a given value.
    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to fill null values.
        value: The value to replace null values with.
    Returns:
        pd.DataFrame: The DataFrame with null values filled.
    """
    dataframe[columns] = dataframe[columns].fillna(value)
    return dataframe


def replace_outliers(df, features):
    """
    Replace outliers using IQR method (capping).
    
    Parameters:
    - df: DataFrame cần xử lý
    - features: List các cột cần xử lý outliers

    Returns:
    - df: DataFrame sau khi đã xử lý outliers
    """
    for feature in features:
        Q1 = df[feature].quantile(0.25)  # Quartile 1
        Q3 = df[feature].quantile(0.75)  # Quartile 3
        IQR = Q3 - Q1  # Interquartile Range
        
        lower_bound = Q1 - 1.5 * IQR  # Upper
        upper_bound = Q3 + 1.5 * IQR  # LowerLower

        #capping
        df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
        df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])

    return df


def count_outliers(df, numerical_features):
    outlier_counts = {}
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_counts[feature] = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
    return outlier_counts