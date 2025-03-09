import rampwf as rw
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error


problem_title = 'SSRD and SP Estimation from Weather Data'

# Global Variables
stations = ['paris', 'brest', 'london', 'marseille', 'berlin']
variables = ['t2m', 'tp', 'skt', 'u10', 'v10', 'd2m', 'blh', 'sp', 'ssrd', 'tcc']

# defining the median values for each variable to use for RMSE normalization
MEDIAN_VARIABLES = {
    "t2m": 42.22989,
    "tp": 0.0014213523,
    "skt": 47.08624,
    "u10": 16.019768,
    "v10": 16.840459,
    "d2m": 37.161500000000046,
    "blh": 1951.12801,
    "sp": 7720.484999999986,
    "ssrd": 1240309.333,
    "tcc": 1.0,
}


# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()

# An object implementing the workflow
workflow = rw.workflows.Estimator()

class RMSE(rw.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name='rmse', precision=4, n_targets=10):
        self.name = name
        self.precision = precision
        self.n_targets = n_targets  # Number of target variables (SSRD & SP)

    def __call__(self, y_true, y_pred):

        length_horizon = len(y_true) // self.n_targets

        # Compute RMSE for each target variable
        errors = []
        for i, var in enumerate(variables):
            normalization_factor = MEDIAN_VARIABLES[var]
            start_idx = length_horizon * i
            end_idx = length_horizon * (i + 1)
            rmse = root_mean_squared_error(y_true[start_idx:end_idx],y_pred[start_idx:end_idx])
            errors.append(rmse / normalization_factor)

        # Return the mean RMSE across all targets
        final_rmse = np.mean(errors)
        
        return final_rmse


# Define score types
score_types = [
    RMSE(name='rmse', precision=4)
]

def get_cv(X, y):
    """
    Time Series Cross-Validation function required by RAMP.
    Each model's corresponding slice of X gets its own TimeSeriesSplit.
    The resulting train-test splits from each model are concatenated.
    """
    num_splits = 2  # Number of splits
    num_models = len(variables)  # Assuming one model per target variable
    length_horizon = X.shape[0] // num_models  # Divide X into chunks

    tscv = TimeSeriesSplit(n_splits=num_splits)  # Time series CV
    splits = []
    for i in range(num_models):
        start_idx = length_horizon * i
        end_idx = length_horizon * (i + 1)

        X_subset = X[start_idx:end_idx]

        # Generate TimeSeriesSplit indices for each slice
        var_split_indices = list(tscv.split(X_subset))
        var_split_indices = [(split_indices[0]+start_idx,split_indices[1]+start_idx) for split_indices in var_split_indices]
        splits.append(var_split_indices)

    for split_idx in range(num_splits):
        # Store the indices
        train_idx_list = [splits[i][split_idx][0] for i in range(num_models)]
        val_idx_list = [splits[i][split_idx][1] for i in range(num_models)]

        # Concatenate all train and val indices across different variable models
        train_idx_concat = np.concatenate(train_idx_list)
        val_idx_concat = np.concatenate(val_idx_list)

        yield train_idx_concat, val_idx_concat



def _read_data(path, df_filename):
    df = pd.read_csv(os.path.join(path, 'data', df_filename), index_col="time")

    _target_variables = [var+"_paris" for var in variables]
    y = df[_target_variables].to_numpy().T.flatten()  # Flatten but still structured correctly

    X = df.drop(_target_variables, axis=1)
    X = np.tile(X, (len(_target_variables), 1))  # Repeat X for each target variable

    return X, y

def get_train_data(path='.'):
    df_filename = 'train.csv'
    return _read_data(path, df_filename)

def get_test_data(path='.'):
    df_filename = 'test.csv'
    return _read_data(path, df_filename)
