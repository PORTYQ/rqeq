"""
Model module for Crypto Forecast Application
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import shap
from config import config
from utils import calculate_metrics, save_model_weights


class TimeSeriesModel(BaseEstimator):
    """Wrapper for Keras models to work with sklearn"""

    def __init__(
        self,
        model_type: str = "LSTM",
        n_layers: int = 2,
        units: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        window_size: int = 60,
        n_features: int = 10,
        horizon: int = 7,
    ):
        self.model_type = model_type
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.n_features = n_features
        self.horizon = horizon
        self.model = None
        self.history = None

    def build_model(self) -> Sequential:
        """Build LSTM or GRU model"""
        model = Sequential()

        # Choose layer type
        LayerType = LSTM if self.model_type == "LSTM" else GRU

        # Add layers
        for i in range(self.n_layers):
            return_sequences = i < self.n_layers - 1

            if i == 0:
                model.add(
                    LayerType(
                        self.units,
                        return_sequences=return_sequences,
                        input_shape=(self.window_size, self.n_features),
                    )
                )
            else:
                model.add(LayerType(self.units, return_sequences=return_sequences))

            model.add(Dropout(self.dropout))

        # Output layer
        model.add(Dense(self.horizon))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TimeSeriesModel":
        """Fit the model"""
        self.model = self.build_model()

        # Split data for validation
        val_split = int(0.8 * len(X))
        X_train, X_val = X[:val_split], X[val_split:]
        y_train, y_val = y[:val_split], y[val_split:]

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )

        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.max_epochs,
            batch_size=config.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility"""
        return {
            "model_type": self.model_type,
            "n_layers": self.n_layers,
            "units": self.units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "window_size": self.window_size,
            "n_features": self.n_features,
            "horizon": self.horizon,
        }

    def set_params(self, **params) -> "TimeSeriesModel":
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ModelTrainer:
    """Class for training and evaluating models"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.best_model = None
        self.best_params = None
        self.training_history = []

    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        n_iter: int = 20,
    ) -> Tuple[Dict[str, Any], float]:
        """Perform random hyperparameter search"""
        self.logger.info(f"Starting hyperparameter search with {n_iter} iterations")

        # Define parameter distributions
        param_dist = {
            "n_layers": list(range(config.hp_layers[0], config.hp_layers[1] + 1)),
            "units": list(range(config.hp_units[0], config.hp_units[1] + 1, 32)),
            "dropout": np.linspace(config.hp_dropout[0], config.hp_dropout[1], 10),
            "learning_rate": np.logspace(
                np.log10(config.hp_lr[0]), np.log10(config.hp_lr[1]), 10
            ),
        }

        best_score = float("inf")
        best_params = None

        for i in range(n_iter):
            # Sample parameters
            params = {
                "model_type": model_type,
                "n_layers": np.random.choice(param_dist["n_layers"]),
                "units": np.random.choice(param_dist["units"]),
                "dropout": np.random.choice(param_dist["dropout"]),
                "learning_rate": np.random.choice(param_dist["learning_rate"]),
                "window_size": X_train.shape[1],
                "n_features": X_train.shape[2],
                "horizon": y_train.shape[1],
            }

            try:
                # Create and train model
                model = TimeSeriesModel(**(params or {}))
                model.fit(X_train, y_train)

                # Evaluate
                val_loss = min(model.history.history["val_loss"])

                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    self.best_model = model

                self.logger.info(f"Iteration {i+1}/{n_iter}: val_loss={val_loss:.4f}")

            except Exception as e:
                self.logger.error(f"Error in iteration {i+1}: {e}")
                continue

        self.best_params = best_params
        return best_params, best_score

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str,
        params: Optional[Dict[str, Any]] = None,
        use_hyperparam_search: bool = True,
    ) -> TimeSeriesModel:
        """Train model with given parameters"""

        if use_hyperparam_search and not params:
            # Perform hyperparameter search
            params, _ = self.hyperparameter_search(X_train, y_train, model_type)
        if not isinstance(params, dict) or params is None:
            self.logger.warning("params %s → подменяю на дефолт", type(params))
            params = {
                "model_type": model_type,
                "n_layers": 2,
                "units": 128,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "window_size": X_train.shape[1],
                "n_features": X_train.shape[2],
                "horizon": y_train.shape[1],
            }

        self.logger.debug("PARAMS passed to TimeSeriesModel → %s", params)
        model = TimeSeriesModel(**params)

        # Train final model
        self.logger.info("Training final model with parameters: %s", params)
        # Combine train and validation for final training
        X_combined = np.concatenate([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        model.fit(X_combined, y_combined)

        # Store training history
        self.training_history = model.history.history

        return model

    def evaluate_model(
        self, model: TimeSeriesModel, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = model.predict(X_test)

        # Calculate metrics for each horizon step
        metrics = {}
        for h in range(y_test.shape[1]):
            h_metrics = calculate_metrics(y_test[:, h], predictions[:, h])
            for metric_name, value in h_metrics.items():
                metrics[f"{metric_name}_h{h+1}"] = value

        # Calculate average metrics
        avg_metrics = calculate_metrics(y_test.flatten(), predictions.flatten())
        for metric_name, value in avg_metrics.items():
            metrics[f"{metric_name}_avg"] = value

        return metrics

    def explain_predictions(
        self,
        model: TimeSeriesModel,
        X_sample: np.ndarray,
        feature_names: List[str],
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for model predictions"""
        self.logger.info("Generating SHAP explanations")

        # Limit samples for performance
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]

        try:
            # Create explainer
            start_time = time.time()

            # Define prediction function for SHAP
            def predict_fn(X):
                # Reshape if needed
                if len(X.shape) == 2:
                    X = X.reshape((X.shape[0], self.best_model.window_size, -1))
                return model.predict(X).mean(axis=1)  # Average over horizon

            # Use KernelExplainer for deep learning models
            explainer = shap.KernelExplainer(
                predict_fn,
                X_sample.reshape((X_sample.shape[0], -1))[
                    :10
                ],  # Use subset as background
            )

            # Calculate SHAP values
            shap_values = explainer.shap_values(
                X_sample.reshape((X_sample.shape[0], -1)), check_additivity=False
            )

            # Check timeout
            if time.time() - start_time > config.shap_timeout:
                self.logger.warning("SHAP calculation timed out")
                return None

            # Create feature names for flattened input
            full_feature_names = []
            for t in range(X_sample.shape[1]):
                for feat in feature_names:
                    full_feature_names.append(f"{feat}_t-{X_sample.shape[1]-t}")

            return {
                "shap_values": shap_values,
                "feature_names": full_feature_names,
                "X_sample": X_sample.reshape((X_sample.shape[0], -1)),
            }

        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {e}")
            return None

    def compare_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        with_macro: bool,
        without_macro: bool,
    ) -> pd.DataFrame:
        """Compare models with and without macro indicators"""
        results = []

        # Train model without macro indicators
        if without_macro:
            self.logger.info("Training model without macro indicators")
            model_no_macro = self.train_model(
                X_train[:, :, :5],  # Only OHLCV features
                y_train,
                X_test[:, :, :5],
                y_test,
                model_type="LSTM",
                use_hyperparam_search=False,
            )

            metrics_no_macro = self.evaluate_model(
                model_no_macro, X_test[:, :, :5], y_test
            )

            results.append(
                {
                    "Model": "LSTM (Price Only)",
                    **{k: v for k, v in metrics_no_macro.items() if "avg" in k},
                }
            )

        # Train model with macro indicators
        if with_macro:
            self.logger.info("Training model with macro indicators")
            model_with_macro = self.train_model(
                X_train,
                y_train,
                X_test,
                y_test,
                model_type="LSTM",
                params=self.best_params or {},
                use_hyperparam_search=False,
            )

            metrics_with_macro = self.evaluate_model(model_with_macro, X_test, y_test)

            results.append(
                {
                    "Model": "LSTM (Price + Macro)",
                    **{k: v for k, v in metrics_with_macro.items() if "avg" in k},
                }
            )

            # Calculate improvement
            if without_macro and with_macro:
                improvement = {
                    "Model": "Improvement (%)",
                    "MAE_avg": (
                        (metrics_no_macro["MAE_avg"] - metrics_with_macro["MAE_avg"])
                        / metrics_no_macro["MAE_avg"]
                        * 100
                    ),
                    "RMSE_avg": (
                        (metrics_no_macro["RMSE_avg"] - metrics_with_macro["RMSE_avg"])
                        / metrics_no_macro["RMSE_avg"]
                        * 100
                    ),
                    "MAPE_avg": (
                        (metrics_no_macro["MAPE_avg"] - metrics_with_macro["MAPE_avg"])
                        / metrics_no_macro["MAPE_avg"]
                        * 100
                    ),
                }
                results.append(improvement)

        return pd.DataFrame(results)
