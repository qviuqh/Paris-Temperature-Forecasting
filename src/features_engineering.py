import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA


def create_lag_features(data, columns, lag):
    """
    Creates lagged features for the specified columns in the dataset.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to create lag features for.
    - lag (int): The number of lag periods to create.

    Returns:
    - pd.DataFrame: DataFrame with lagged features added.
    """
    for col in columns:
        for i in range(1, lag + 1):
            data[f"{col}_lag_{i}"] = data[col].shift(i).bfill()
    return data

def create_rolling_features(data, columns, window):
    """
    Creates rolling features for the specified columns in the dataset.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to create rolling features for.
    - window (int): The window size for the rolling calculation.

    Returns:
    - pd.DataFrame: DataFrame with rolling features added.
    """
    for col in columns:
        data[f"{col}_rolling_mean_{window}"] = data[col].rolling(window=window).mean()
        data[f"{col}_rolling_std_{window}"] = data[col].rolling(window=window).std()
    # Find the null values in rolling mean and rolling std columns
    rolling_cols = [col for col in data.columns if "rolling_mean" in col or "rolling_std" in col]
    # Fill NaN with the average value of each column
    data[rolling_cols] = data[rolling_cols].fillna(data[rolling_cols].mean())

    return data


def create_specific_features(df):
    """
    Creates specific features for the specified columns in the dataset.
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    Returns:
    - pd.DataFrame: DataFrame with specific features added.
    """
    df["temp_range"] = df["tempmax"] - df["tempmin"] # Daily temperature fluctuation
    df["dew_spread"] = df["temp"] - df["dew"] # Difference between temperature and dew point
    df["high_humidity"] = (df["humidity"] > 80).astype(int) # Create high_humidity variable if humidity is above 80% (usually causes rain/fog)
    df["rain_intensity"] = df["precip"] / (df["precipcover"] + 1e-5) # Rainfall intensity during the day
    df["binary_rain"] = (df["precip"] > 0).astype(int) # Check if a day will rain or not
    
    df['heat_index'] = -8.7847 + 1.6114 * df['temp'] + 2.3385 * df['humidity'] - 0.1461 * df['temp'] * df['humidity']
    df['wind_chill'] = 13.12 + 0.6215 * df['temp'] - 11.37 * df['windspeed']**0.16 + 0.3965 * df['temp'] * df['windspeed']**0.16
    df['cloud_wind_effect'] = (df['cloudcover'] * df['windspeed']) / 100
    df['pressure_temp_index'] = df['sealevelpressure'] * df['temp']
    df['humidity_cloud_index'] = (df['humidity'] * df['cloudcover']) / 100
    df['solar_temp_index'] = df['solarradiation'] * df['temp']
    df['solar_wind_effect'] = df['solarradiation'] / (df['windspeed'] + 1)
    df['uv_cloud_index'] = df['uvindex'] * (1 - df['cloudcover'] / 100)
    df['dew_temp_index'] = df['dew'] * df['temp']
    df['cloud_radiation_index'] = df['cloudcover'] * df['solarradiation']
    df['wind_temp_index'] = df['windspeed'] * df['temp']
    df['pressure_cloud_index'] = df['sealevelpressure'] * (1 - df['cloudcover'] / 100)
    df['uv_humidity_index'] = df['uvindex'] * df['humidity']
    df["temp_humidity_index"] = df["temp"] * df["humidity"]
    df["wind_precip_index"] = df["windspeed"] * df["precip"]

    # Wind direction grouping
    def categorize_wind_direction(degree):
        if 0 <= degree < 90:
            return "Northeast"
        elif 90 <= degree < 180:
            return "Southeast"
        elif 180 <= degree < 270:
            return "Southwest"
        else:
            return "Northwest"
        
    df["wind_category"] = df["winddir"].apply(categorize_wind_direction)
    # Difference between windgust and windspeed
    df["wind_variability"] = df["windgust"] - df["windspeed"] # High means that strong wind
    # Cloud Cover Index
    df["cloudiness_level"] = pd.cut(df["cloudcover"], bins=[0, 30, 70, 100], labels=["Clear", "Partly Cloudy", "Overcast"])
    # Fog prediction based on visibility
    df["foggy"] = (df["visibility"] < 2).astype(int)
    # UV hazard level
    df["high_uv"] = (df["uvindex"] > 6).astype(int)
    # Create time-based features
    df['month'] = df.index.month
    # Separate the day, month, year, and season components
    df["season"] = df["month"].apply(lambda x: "winter" if x in [12,1,2] else 
                                                "spring" if x in [3,4,5] else
                                                "summer" if x in [6,7,8] else "autumn")
    df.drop(columns=["month"], axis=1, inplace=True)
    # The time between sunrise and sunset
    df["daylength (hours)"] = (df["sunset"] - df["sunrise"]).dt.total_seconds() / 3600
    # Label encoding 'conditions'
    le = LabelEncoder()
    df["conditions_encoded"] = le.fit_transform(df["conditions"])
    df["cloudiness_level_encoded"] = le.fit_transform(df["cloudiness_level"])
    # One-Hot Encoding 'preciptype', 'wind_category' và 'season'
    preciptype_one_hot = df["preciptype"].str.get_dummies(sep=",")
    preciptype_one_hot.columns = [f"preciptype_{col}" for col in preciptype_one_hot.columns]
    df = pd.concat([df, preciptype_one_hot], axis=1).drop(columns=["preciptype"], axis=1)
    df = pd.get_dummies(df, columns=['wind_category', 'season'], drop_first=True)

