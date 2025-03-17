import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import OneHotEncoder


def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Data Overview:")
    print(df.head())
    print("Data Info:")
    print(df.info())
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth 
    using the Haversine formula. Returns distance in kilometers.
    """
    R = 6371  # Radius of Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def one_hot_encode_columns(df, columns, encoder=None):
    """
    One-hot encode the specified columns using scikit-learn's OneHotEncoder.
    If no encoder is provided, a new one is created and fit on the data.
    """
    if encoder is None:
        # Use sparse_output=False instead of sparse
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
    else:
        encoded = encoder.transform(df[columns])
        
    # Create a DataFrame with the new one-hot encoded columns
    encoded_df = pd.DataFrame(
        encoded, 
        columns=encoder.get_feature_names_out(columns), 
        index=df.index
    )
    # Drop original categorical columns and join the encoded columns
    df = df.drop(columns=columns)
    df = pd.concat([df, encoded_df], axis=1)
    return df, encoder



def preprocess_data(df):
    # dropping unnecessary column 0
    if "Unnamed: 0" in df.columns:
        df.drop(columns = ['Unnamed: 0'], inplace = True)

    # Converting transaction date/time to datetime object
    df["trans_date_trans_time"] = pd.to_datetime(df['trans_date_trans_time'])

    # time - based features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    # user's age at time of transaction time
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors = "coerce")
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    else:
        df['age'] = np.nan

    # nCalculate distance between user lat/long and merchant lat/long
    df['distance_km'] = df.apply(lambda row: haversine_distance(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)

    # Rename 'amt' to 'amount' for clarity
    if 'amt' in df.columns:
        df.rename(columns={'amt': 'amount'}, inplace=True)

    #  Handle missing values (example: fill numeric columns with 0) (there are no null values but still a good practice to include this)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    #One-hot encode 'category' and 'gender' using OneHotEncoder with handle_unknown='ignore'
    df, encoder = one_hot_encode_columns(df, ['category', 'gender'])

    return df


if __name__ == "__main__":
    filepath = 'data/fraudTest.csv'
    df = load_data(filepath)
    df = preprocess_data(df)
    print(df.head())
    print(df.info())
    df.to_csv('data/preprocessed_transactions.csv', index=False)
    print("Preprocessing complete. Data saved to data/preprocessed_transactions.csv")

