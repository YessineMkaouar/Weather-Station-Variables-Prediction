import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Global Variables
stations = ['paris', 'brest', 'london', 'marseille', 'berlin']
variables = ['t2m', 'tp', 'skt', 'u10', 'v10', 'd2m', 'blh', 'sp', 'ssrd', 'tcc']

####################################
# Multi-Model Regressor for Paris Variables with Feature Selection
####################################

class MultiTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, top_k_features=10):
        self.n_targets = len(variables)
        self.top_k_features = top_k_features  # Number of top correlated features
        self.selected_features = {}  # Store selected features per target
        self.models = {
            "t2m_paris": LinearRegression(),
            "tp_paris": XGBRegressor(
                                    objective='reg:squarederror',
                                    n_estimators=300,
                                    learning_rate=0.1,
                                    max_depth=5,
                                    random_state=42,
                                    n_jobs=-1,
                                    verbosity=0
                                ),
            "skt_paris": LinearRegression(),
            "u10_paris": LinearRegression(),
            "v10_paris": LinearRegression(),
            "d2m_paris": LinearRegression(),
            "blh_paris": XGBRegressor(
                                    objective='reg:squarederror',
                                    n_estimators=100,
                                    learning_rate=0.05,
                                    max_depth=7,
                                    random_state=42,
                                    n_jobs=-1,
                                    verbosity=0
                                ),
            "sp_paris": RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, 
                                               min_samples_leaf=3, max_features="sqrt", random_state=42),
            "ssrd_paris": XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=5, 
                                        learning_rate=0.1, colsample_bytree=0.8, subsample=1.0, 
                                        random_state=42, n_jobs=-1, verbosity=0),
            "tcc_paris": XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=100,
                            learning_rate=0.05,
                            max_depth=7,
                            random_state=42,
                            n_jobs=-1,
                            verbosity=0
                        )
        }
    
    def _select_top_k_features(self, X, y_target, top_k=10):
        """
        Selects the top K features most correlated with the target variable.
        """
        if not isinstance(X, pd.DataFrame):  # Convert X to DataFrame if it's a NumPy array
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        correlations = X.corrwith(pd.Series(y_target)).abs().sort_values(ascending=False)
        return correlations.index[:top_k].tolist()

    def fit(self, X, y):
        self.selected_features = {}
        self.scalers = {}  # Store scalers for each variable

        # Convert X to DataFrame
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        length_horizon = len(X) // len(self.models)
        
        for i, target_var in enumerate(list(self.models.keys())):
            start_idx = length_horizon * i
            end_idx = length_horizon * (i + 1)

            X_var = X.iloc[start_idx:end_idx]
            y_var = y[start_idx:end_idx]

            # Select top correlated features AFTER scaling
            self.selected_features[target_var] = self._select_top_k_features(X_var, y_var)

            # Ensure X is a DataFrame before selecting features
            X_selected = X_var[self.selected_features[target_var]]

            self.models[target_var].fit(X_selected, y_var)

        return self

    def predict(self, X):

        # Convert X to DataFrame if it's a NumPy array
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        predictions = []
        length_horizon = len(X) // len(self.models)

        for i, target_var in enumerate(list(self.models.keys())):

            start_idx = length_horizon * i
            end_idx = length_horizon * (i + 1)
            X_var = X.iloc[start_idx:end_idx]

            # Select the top features used in training
            X_selected = X_var[self.selected_features[target_var]]
            pred = self.models[target_var].predict(X_selected)
            predictions.append(pred)
            (f"mean prediction of {target_var} is {np.mean(pred)}")

        return np.concatenate(predictions)  # 1D array



####################################
# Pipeline Construction
####################################

def get_estimator():
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # ('scaler', StandardScaler())
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', numeric_pipeline),
        ('regressor', MultiTargetRegressor(top_k_features=10))  # Select top 10 correlated features
    ])
    
    return pipeline
