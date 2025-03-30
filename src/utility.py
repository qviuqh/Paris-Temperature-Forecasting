# Description: Utility functions for data preprocessing and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import phik


def nan_df_create(data):
    nan_percentages = data.isna().sum() * 100 / len(data)
    df_nan = pd.DataFrame({'column': nan_percentages.index, 'percent': nan_percentages.values})
    df_nan.sort_values(by='percent', ascending=False, inplace=True)
    return df_nan


def plot_outlier_boxplots(data, numerical_features, log_scale=False):
    num_features = len(numerical_features)
    rows = (num_features // 3) + 1  # Arrange in rows of 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    
    axes = axes.flatten()  # Flatten for easy iteration
    for i, feature in enumerate(numerical_features):
        sns.boxplot(y=data[feature], ax=axes[i], palette="coolwarm", width=0.5)
        axes[i].set_title(f"Boxplot of {feature}", fontsize=12)
        axes[i].set_ylabel("")
        if log_scale:
            axes[i].set_yscale("log")  # Log scale for skewed distributions
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_nan_percent(df_nan, title_name, tight_layout=True, figsize=(20,8), grid=False, rotation=90):
    if df_nan.percent.sum() != 0:
        print(f"Number of columns with NaN values: {df_nan[df_nan['percent'] != 0].shape[0]}")
        plt.figure(figsize=figsize, tight_layout=tight_layout)
        sns.barplot(x='column', y='percent', data=df_nan[df_nan['percent'] > 0])
        plt.xticks(rotation=rotation)
        plt.xlabel('Column Name')
        plt.ylabel('Percentage of NaN values')
        plt.title(f'Percentage of NaN values in {title_name}')
        if grid:
            plt.grid()
        plt.show()
    else:
        print(f"The dataframe {title_name} does not contain any NaN values.")


def plot_outlier_boxplots(data, numerical_features, log_scale=False):
    outlier_features = []
    
    # Detect features with outliers using IQR
    for feature in numerical_features:
        Q1 = np.percentile(data[feature].dropna(), 25)
        Q3 = np.percentile(data[feature].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if any((data[feature] < lower_bound) | (data[feature] > upper_bound)):
            outlier_features.append(feature)
    
    # If no features have outliers, return early
    if not outlier_features:
        print("No features contain outliers.")
        return
    
    num_features = len(outlier_features)
    rows = (num_features // 3) + (num_features % 3 > 0)  # Adjust rows dynamically
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # Ensure axes is always a list for consistent handling
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, feature in enumerate(outlier_features):
        sns.boxplot(y=data[feature], ax=axes[i], width=0.5)
        axes[i].set_title(f"Boxplot of {feature}", fontsize=12)
        axes[i].set_ylabel("")
        if log_scale:
            axes[i].set_yscale("log")  # Apply log scale if specified
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def nan_df_create(data):
    nan_percentages = data.isna().sum() * 100 / len(data)
    df_nan = pd.DataFrame({'column': nan_percentages.index, 'percent': nan_percentages.values})
    df_nan.sort_values(by='percent', ascending=False, inplace=True)
    return df_nan


def plot_nan_percent(df_nan, title_name, tight_layout=True, figsize=(20,8), grid=False, rotation=90):
    if df_nan.percent.sum() != 0:
        print(f"Number of columns with NaN values: {df_nan[df_nan['percent'] != 0].shape[0]}")
        plt.figure(figsize=figsize, tight_layout=tight_layout)
        sns.barplot(x='column', y='percent', data=df_nan[df_nan['percent'] > 0])
        plt.xticks(rotation=rotation)
        plt.xlabel('Column Name')
        plt.ylabel('Percentage of NaN values')
        plt.title(f'Percentage of NaN values in {title_name}')
        if grid:
            plt.grid()
        plt.show()
    else:
        print(f"The dataframe {title_name} does not contain any NaN values.")


def plot_phik_matrix(data, categorical_columns, figsize=(20,20), mask_upper=True, tight_layout=True, linewidth=0.1, fontsize=10, cmap='Blues'):
    data_for_phik = data[categorical_columns].astype('object')
    phik_matrix = data_for_phik.phik_matrix()
    mask_array = np.triu(np.ones(phik_matrix.shape)) if mask_upper else np.zeros(phik_matrix.shape)
    plt.figure(figsize=figsize, tight_layout=tight_layout)
    sns.heatmap(phik_matrix, annot=False, mask=mask_array, linewidth=linewidth, cmap=cmap)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.title("Phi-K Correlation Heatmap for Categorical Features")
    plt.show()


def check_stationary(series, window=7):
    """"
    Check the stationarity of a time series by:
    1. Plotting rolling mean and rolling std
    2. Running ADF Test
    3. Printing the test result

    Parameters:
    - series: Pandas Series, the data series to be tested
    - window: int, rolling mean & std window size (default = 7)

    Returns:
    - None (only display the plot and print the result)
    """

    # Calculate rolling mean and rolling std
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Plot the plot
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Original Data", color="#073b4c", alpha=0.5)
    plt.plot(rolling_mean, label="Rolling Mean", color="#ef476f")
    plt.plot(rolling_std, label="Rolling Std", color="#ffd166")
    plt.legend()
    plt.title("Rolling Mean & Standard Deviation")
    plt.show()

    # ADF Test
    result = adfuller(series.dropna())

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])

    # Conclusion
    if result[1] < 0.05:
        print("✅ Data is stationary (can use model immediately).")
    else:
        print("⚠️ Data is not stationary (needs to be processed before forecasting).")


def make_stationary(series, max_diff=5, significance_level=0.05):
    """
    Tự động lấy sai phân cho đến khi chuỗi trở nên dừng.
    
    Args:
    - series: pandas Series, chuỗi thời gian cần kiểm tra.
    - max_diff: Số lần lấy sai phân tối đa.
    - significance_level: Ngưỡng ý nghĩa cho kiểm định ADF.
    
    Returns:
    - stationary_series: Chuỗi đã được làm dừng.
    - num_diffs: Số lần lấy sai phân cần thiết.
    - adf_results: Kết quả kiểm định ADF sau khi dừng.
    """
    num_diffs = 0
    current_series = series.copy()

    while num_diffs < max_diff:
        # Kiểm định Dickey-Fuller
        adf_test = adfuller(current_series.dropna())
        adf_statistic, p_value, _, _, critical_values, _ = adf_test

        # Nếu p-value < 0.05, chuỗi đã dừng
        if p_value < significance_level:
            return current_series, num_diffs, {
                "ADF Statistic": adf_statistic,
                "p-value": p_value,
                "Critical Values": critical_values
            }

        # Nếu chưa dừng, tiếp tục lấy sai phân
        current_series = current_series.diff().dropna()
        num_diffs += 1

    # Nếu vượt quá max_diff mà vẫn không dừng
    return current_series, num_diffs, {
        "ADF Statistic": adf_statistic,
        "p-value": p_value,
        "Critical Values": critical_values
    }


class correlation_matrix:
    def __init__(self, data, columns_to_drop, figsize=(25,23), mask_upper=True, tight_layout=True,
                 linewidth=0.1, fontsize=10, cmap='Blues'):
        self.data = data
        self.columns_to_drop = columns_to_drop
        self.figsize = figsize
        self.mask_upper = mask_upper
        self.tight_layout = tight_layout
        self.linewidth = linewidth
        self.fontsize = fontsize
        self.cmap = cmap
    
    def plot_correlation_matrix(self):
        numerical_data = self.data.drop(self.columns_to_drop, axis=1).select_dtypes(include=np.number)
        self.corr_data = numerical_data.corr()
        mask_array = np.triu(np.ones(self.corr_data.shape)) if self.mask_upper else np.zeros(self.corr_data.shape)
        plt.figure(figsize=self.figsize, tight_layout=self.tight_layout)
        sns.heatmap(self.corr_data, annot=False, mask=mask_array, linewidth=self.linewidth, cmap=self.cmap)
        plt.xticks(rotation=90, fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.title("Correlation Heatmap for Numerical Features")
        plt.show()