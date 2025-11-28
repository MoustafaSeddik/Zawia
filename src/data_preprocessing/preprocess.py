#%% import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, PowerTransformer,
                                   RobustScaler, MinMaxScaler)
import os
from src.utils.genPlots import BoxPlot_data
from colorama import Fore, Style
import torch
#%% Load the data

def load_datasets(trainfilename, targetfilename, train_size, val_size):
    # check if both filenames exist and the directory exists

    for filename in [trainfilename, targetfilename]:
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.isdir(dir_path):
            print("*********************ERROR*****************************")
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
        if not os.path.isfile(filename):
            print("*********************ERROR******************************")
            raise FileNotFoundError(f"File does not exist: {filename}")

    # Load datasets
    X_train = np.load(trainfilename, allow_pickle=True)
    y_train = np.load(targetfilename, allow_pickle=True)

    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train)

    columns = [f"sph_{x}" for x in range(X_train.shape[1])]
    if X_train.shape[1] == 3 or "X3" in trainfilename:
        columns[0] = "zen"
        columns[1] = "azi"
        columns[2] = "uFac"
    elif X_train.shape[1] == 4 or "X4" in trainfilename:
        columns[0] = "x"
        columns[1] = "y"
        columns[2] = "z"
        columns[3] = "uFac"

    X_train_df.columns = columns

    y_train_df.columns = ["F_1"]

    train = pd.concat([X_train_df, y_train_df], axis=1)
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data into train, validation and test sets
    train, val_test = train_test_split(train, train_size=train_size,
                                       random_state=42, shuffle=True)
    test_size = val_size / (1 - train_size)
    val, test = train_test_split(val_test, test_size=test_size,
                                 random_state=42, shuffle=False)

    # Check the shape of the data
    print(f"\n{Fore.RED}Shape of df_train:{Style.RESET_ALL}{train.shape}")
    print(f"{Fore.RED}Shape of df_val:{Style.RESET_ALL}{test.shape}")
    print(f"{Fore.RED}Shape of df_test:{Style.RESET_ALL}{val.shape}")

    return train, val, test

# %% Preparing the Data

def prepare_data(df_train, df_val, df_test,
                 normalization='StandardScaler', visualization="Training"):
    """
    Prepare data by scaling features and labels using a specified normalization method.

    Args:
        df_train (pd.DataFrame): Training dataset
        df_val (pd.DataFrame): Validation dataset
        df_test (pd.DataFrame): Test dataset
        normalization (str): Normalization method to use
            Options: 'StandardScaler', 'PowerTransformer', 'RobustScaler', 'MinMaxScaler'

    Returns:
        tuple: Scaled and transformed data as PyTorch tensors
    """

    # Split features and target
    X_train_df = df_train.drop(columns=['F_1'])
    y_train_df = df_train['F_1']

    X_val_df = df_val.drop(columns=['F_1'])
    y_val_df = df_val['F_1']

    X_test_df = df_test.drop(columns=['F_1'])
    y_test_df = df_test['F_1']

    # Convert to numpy arrays
    X_train_raw = X_train_df.values
    y_train_raw = y_train_df.values

    X_val_raw = X_val_df.values
    y_val_raw = y_val_df.values

    X_test_raw = X_test_df.values
    y_test_raw = y_test_df.values

    # Dictionary of available scalers
    scalers = {
        'standardScaler': (StandardScaler(),
                           StandardScaler()),
        'powerTransformer': (
            PowerTransformer(method='yeo-johnson', standardize=True),
            PowerTransformer(method='yeo-johnson', standardize=True)
        ),
        'robustScaler': (
            RobustScaler(with_centering=True, with_scaling=True),
            RobustScaler(with_centering=True, with_scaling=True)
        ),
        'minMaxScaler': (MinMaxScaler(),
                         MinMaxScaler()),
        'None': (None, None)
    }

    # Get scalers or raise error if normalization method not found
    try:
        scaler_X, scaler_y = scalers[normalization]
    except KeyError:
        raise ValueError(
            f"Unsupported normalization method: {normalization}. "
            f"Available methods: {list(scalers.keys())}"
        )

    # Apply scaling if scalers are provided
    if scaler_X is not None and scaler_y is not None:
        # Scale feature data
        X_train = scaler_X.fit_transform(X_train_raw)
        X_val = scaler_X.transform(X_val_raw)
        X_test = scaler_X.transform(X_test_raw)

        # Scale target labels
        y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
        y_val = scaler_y.transform(y_val_raw.reshape(-1, 1))
        y_test = scaler_y.transform(y_test_raw.reshape(-1, 1))
    else:
        # Use raw data if no scaling is requested
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw
        y_train = y_train_raw.reshape(-1, 1)
        y_val = y_val_raw.reshape(-1, 1)
        y_test = y_test_raw.reshape(-1, 1)

    fig = None
    if visualization == "Training":
        fig = BoxPlot_data(X_train, X_train_raw, y_train, y_train_raw,
                     scaling=normalization, datatype="Training")
    elif visualization == "Validation":
        fig = BoxPlot_data(X_val, X_val_raw, y_val, y_val_raw,
                     scaling=normalization, datatype="Validation")
    elif visualization == "Test":
        fig = BoxPlot_data(X_test, X_test_raw, y_test, y_test_raw,
                     scaling=normalization, datatype="Test")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    testData = {
        "X_test_raw": X_test_raw,
        "X_test_df": X_test_df,
        "y_test_df": y_test_df,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    }
    # Check the shape of the data
    print(f"{Fore.RED}Shape of X_train:{Style.RESET_ALL}{X_train.shape}")
    print(f"{Fore.RED}Shape of y_train:{Style.RESET_ALL}{y_train.shape}")

    print(f"{Fore.RED}Shape of X_val:{Style.RESET_ALL}{X_val.shape}")
    print(f"{Fore.RED}Shape of y_val:{Style.RESET_ALL}{y_val.shape}")

    print(f"{Fore.RED}Shape of X_test:{Style.RESET_ALL}{X_test.shape}")
    print(f"{Fore.RED}Shape of y_test:{Style.RESET_ALL}{y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, testData, fig

