from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

from src.models.base_model import BaseModel, ModelMetrics


class LGBMModel(BaseModel):
    """
    LightGBM classifier with sensible defaults for financial data.
    
    Handles class imbalance, categorical features, and provides
    early stopping to prevent overfitting.
    
    Parameters
    ----------
    model_params : Dict[str, Any], optional
        LightGBM parameters. Common parameters:
        - learning_rate : float (default 0.1)
        - num_leaves : int (default 100)
        - min_data_in_leaf : int (default 100)
        - num_iterations : int (default 300)
        - max_depth : int (default -1, no limit)
    early_stopping_rounds : int, optional
        Stop if validation metric doesn't improve for N rounds
    verbose : int, optional
        Verbosity level (-1 for silent, 0 for warnings, >0 for info)
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = -1
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required but not installed")
        
        default_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'num_leaves': 100,
            'min_data_in_leaf': 100,
            'num_iterations': 300,
            'max_bin': 255,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': verbose
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(model_params=default_params)
        
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.training_history_ = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        categorical_features: Optional[list] = None,
        **kwargs
    ) -> 'LGBMModel':
        """
        Fit LightGBM model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training target (binary: 0 or 1)
        X_val : pd.DataFrame, optional
            Validation features
        y_val : np.ndarray, optional
            Validation target
        categorical_features : list, optional
            Names of categorical features
            
        Returns
        -------
        self : LGBMModel
        """
        logger.info("Training LightGBM model...")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        
        if X_val is not None and y_val is not None:
            logger.info(f"Validation samples: {len(X_val)}")
        
        self.feature_names_ = list(X.columns)
        
        self._validate_target(y)
        
        self.model = LGBMClassifier(**self.model_params)
        
        fit_params = {}
        
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['eval_metric'] = ['binary_logloss', 'auc']
            
            if self.early_stopping_rounds > 0:
                logger.info(f"Early stopping enabled with {self.early_stopping_rounds} rounds")
                # LightGBM 4.0+ uses callbacks differently
                try:
                    from lightgbm import early_stopping
                    fit_params['callbacks'] = [early_stopping(self.early_stopping_rounds)]
                except ImportError:
                    # Fallback for older LightGBM versions
                    fit_params['early_stopping_rounds'] = self.early_stopping_rounds
        
        if categorical_features:
            fit_params['categorical_feature'] = categorical_features
            logger.info(f"Using {len(categorical_features)} categorical features")
        
        self.model.fit(X, y, **fit_params)
        
        self.is_fitted_ = True
        
        if hasattr(self.model, 'best_iteration_'):
            logger.info(f"Best iteration: {self.model.best_iteration_}")
        
        if hasattr(self.model, 'best_score_'):
            logger.info(f"Best validation score: {self.model.best_score_}")
        
        train_metrics = self._compute_training_metrics(X, y)
        logger.info(f"Training accuracy: {train_metrics['accuracy']:.4f}")
        
        if X_val is not None and y_val is not None:
            val_metrics = self._compute_validation_metrics(X_val, y_val)
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation log-loss: {val_metrics['log_loss']:.4f}")
            self.training_history_['validation_metrics'] = val_metrics
        
        self.training_history_['training_metrics'] = train_metrics
        
        logger.info("Model training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        self._validate_features(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        np.ndarray
            Predicted probabilities, shape (n_samples, 2)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        self._validate_features(X)
        
        return self.model.predict_proba(X)
    
    def _extract_feature_importance(self) -> Optional[np.ndarray]:
        """Extract feature importance from LightGBM model."""
        if not self.is_fitted_:
            return None
        
        return self.model.feature_importances_
    
    def _validate_target(self, y: np.ndarray) -> None:
        """Validate target is binary."""
        unique_values = np.unique(y)
        
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(
                f"Target must be binary (0, 1). Found: {unique_values}"
            )
        
        if len(unique_values) < 2:
            raise ValueError(
                "Target must have both classes. Found only one class."
            )
        
        class_counts = np.bincount(y)
        logger.info(
            f"Class distribution - 0: {class_counts[0]:,} ({class_counts[0]/len(y):.1%}), "
            f"1: {class_counts[1]:,} ({class_counts[1]/len(y):.1%})"
        )
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """Validate features match training."""
        if self.feature_names_ is None:
            return
        
        missing_features = set(self.feature_names_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        extra_features = set(X.columns) - set(self.feature_names_)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
    
    def _compute_training_metrics(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics on training data."""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        return ModelMetrics.calculate_binary_metrics(y, y_pred, y_pred_proba)
    
    def _compute_validation_metrics(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics on validation data."""
        y_pred = self.predict(X_val)
        y_pred_proba = self.predict_proba(X_val)
        
        return ModelMetrics.calculate_binary_metrics(y_val, y_pred, y_pred_proba)
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history and metrics.
        
        Returns
        -------
        Dict[str, Any]
            Training and validation metrics
        """
        return self.training_history_.copy()


class LGBMModelFactory:
    """Factory for creating LightGBM models with preset configurations."""
    
    @staticmethod
    def create_fast_model() -> LGBMModel:
        """
        Create model optimized for speed.
        
        Suitable for quick experiments and hyperparameter search.
        """
        params = {
            'learning_rate': 0.2,
            'num_leaves': 31,
            'min_data_in_leaf': 50,
            'num_iterations': 100,
            'max_bin': 127
        }
        
        return LGBMModel(model_params=params, early_stopping_rounds=20)
    
    @staticmethod
    def create_accurate_model() -> LGBMModel:
        """
        Create model optimized for accuracy.
        
        Suitable for final model training with more iterations.
        """
        params = {
            'learning_rate': 0.05,
            'num_leaves': 200,
            'min_data_in_leaf': 200,
            'num_iterations': 1000,
            'max_bin': 511,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9
        }
        
        return LGBMModel(model_params=params, early_stopping_rounds=100)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> LGBMModel:
        """
        Create model from configuration dictionary.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration with 'model_params' and optionally 'early_stopping_rounds'
            
        Returns
        -------
        LGBMModel
        """
        model_params = config.get('model_params', {})
        early_stopping = config.get('early_stopping_rounds', 50)
        
        return LGBMModel(
            model_params=model_params,
            early_stopping_rounds=early_stopping
        )
