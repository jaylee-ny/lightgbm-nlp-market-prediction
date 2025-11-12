from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from loguru import logger


class BaseModel(ABC):
    """
    Abstract base class for all predictive models.
    
    All models must implement fit, predict, and predict_proba methods.
    Provides common functionality for saving, loading, and feature importance.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize model with parameters.
        
        Parameters
        ----------
        model_params : Dict[str, Any], optional
            Model-specific hyperparameters
        """
        self.model_params = model_params or {}
        self.model = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Fit model to training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training target
        X_val : pd.DataFrame, optional
            Validation features for early stopping
        y_val : np.ndarray, optional
            Validation target
            
        Returns
        -------
        self : BaseModel
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on
            
        Returns
        -------
        np.ndarray
            Probability predictions, shape (n_samples, n_classes)
        """
        pass
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns
        -------
        pd.DataFrame
            Feature names and importance scores, sorted by importance
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self._extract_feature_importance()
        
        if importance is None:
            logger.warning("Feature importance not available for this model")
            return pd.DataFrame(columns=['feature', 'importance'])
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    @abstractmethod
    def _extract_feature_importance(self) -> Optional[np.ndarray]:
        """
        Extract feature importance from fitted model.
        
        Returns
        -------
        np.ndarray or None
            Feature importance scores
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Path to save model file
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_params': self.model_params,
                'feature_names': self.feature_names_,
                'is_fitted': self.is_fitted_
            }, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Path to model file
            
        Returns
        -------
        self : BaseModel
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model']
        self.model_params = saved_data['model_params']
        self.feature_names_ = saved_data['feature_names']
        self.is_fitted_ = saved_data['is_fitted']
        
        logger.info(f"Model loaded from {load_path}")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns
        -------
        Dict[str, Any]
            Model parameters
        """
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Parameters
        ----------
        **params
            Model parameters to update
            
        Returns
        -------
        self : BaseModel
        """
        self.model_params.update(params)
        return self


class ModelMetrics:
    """Calculate and store model performance metrics."""
    
    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate metrics for binary classification.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            log_loss
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) == 2:
                y_pred_proba = y_pred_proba[:, 1]
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = np.nan
            
            try:
                if len(y_pred_proba.shape) == 1:
                    y_pred_proba_2d = np.vstack([1 - y_pred_proba, y_pred_proba]).T
                else:
                    y_pred_proba_2d = y_pred_proba
                
                metrics['log_loss'] = log_loss(y_true, y_pred_proba_2d)
            except ValueError:
                metrics['log_loss'] = np.nan
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]) -> None:
        """Pretty print metrics."""
        logger.info("Model Performance Metrics:")
        logger.info("-" * 40)
        for metric, value in metrics.items():
            if not np.isnan(value):
                logger.info(f"{metric:15s}: {value:.4f}")
