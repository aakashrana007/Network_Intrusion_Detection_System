import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_pipeline(df):
    """
    Preprocess the dataset for the NIDS model.
        Parameters:
        df (pd.DataFrame): Input dataframe with raw data.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for the model.
    """
    # 1. Remove erroneous 'Label' rows (if any)
    df = df[df["Label"] != "Label"]

    # 2. Ensure 'Protocol' is treated as categorical and create dummies
    df["Protocol"] = df["Protocol"].astype(str)
    protocol_dummies = pd.get_dummies(df["Protocol"], prefix="Protocol")
    df = pd.concat([df, protocol_dummies], axis=1)
    df.drop("Protocol", axis=1, inplace=True)

    # 3. Drop unnecessary columns
    columns_to_drop = [
        'Flow ID',       # Flow identifier
        'Timestamp',      # Timestamp
        'Source IP',      # Source IP (may be irrelevant for modeling)
        'Destination IP',  # Destination IP (may be irrelevant)
        'Dst Port',       # Might not be relevant for the model
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    ]
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors="ignore")

    # 4. Replace inf/-inf with NaN, fill with median or mean)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # 5. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 6. Ensure all columns are numeric
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)  # Replace any remaining NaN values with 0

    # 7. Encode the 'Label' column for multi-class classification
    label_mapping = {
        'BENIGN': 0,           # Normal traffic
        'DoS': 1,              # Denial of Service
        'Bot': 2,              # Botnet attack
        'BruteForce': 3,       # Brute Force
        'Web Attack': 4,       # Web Attack
        'FTP-Patator': 5,      # FTP Patator
        'SSH-Patator': 6,      # SSH Patator
        'Infiltration': 7,     # Infiltration
        'Heartbleed': 8,       # Heartbleed attack
        'DoS GoldenEye': 9,    # DoS GoldenEye
        'DoS Hulk': 10,        # DoS Hulk
        'DoS slowloris': 11,   # DoS Slowloris
        'DoS Slowhttptest': 12,  # DoS Slowhttptest
        'Botnet': 13,          # Botnet attack
        'PortScan': 14,        # Port Scan
        'DDoS': 15,            # DDoS
        'Brute Force XSS': 16,   # Brute Force XSS
        'Brute Force Web': 17,   # Brute Force Web
        'SQL Injection': 18,   # SQL Injection
    }
    # Map the labels in the dataset using the label_mapping
    df["Label"] = df["Label"].map(label_mapping)

    # 8. Move 'Label' column to the end
    label_col = df.pop("Label")
    df["Label"] = label_col

    # 9. Standardize numerical columns (optional)
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


joblib.dump(preprocess_pipeline, 'preprocessor.pkl')
