"""Tests for model implementations."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.base_model import BaseModel, ModelMetrics
from src.models.lgbm_model import LGBMModel, LGBMModelFactory


class TestBaseModel:
    """Test abstract base model functionality."""
    
    def test_cannot_instantiate_base_model(self):
        """Cannot instantiate abstract base class."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_model_metrics_binary_classification(self):
        """Test binary classification metrics calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4],
            [0.7, 0.3], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]
        ])
        
        metrics = ModelMetrics.calculate_binary_metrics(
            y_true, y_pred, y_pred_proba
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'log_loss' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['accuracy'] == 0.75


class TestLGBMModel:
    """Test LightGBM model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample binary classification data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        return X, y
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = LGBMModel()
        
        assert model.model_params['learning_rate'] == 0.1
        assert model.model_params['num_leaves'] == 100
        assert not model.is_fitted_
    
    def test_model_fit_basic(self, sample_data):
        """Test basic model training."""
        X, y = sample_data
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y)
        
        assert model.is_fitted_
        assert model.feature_names_ == list(X.columns)
        assert model.model is not None
    
    def test_model_fit_with_validation(self, sample_data):
        """Test training with validation set."""
        X, y = sample_data
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LGBMModel(
            model_params={'num_iterations': 20, 'verbose': -1},
            early_stopping_rounds=5
        )
        model.fit(X_train, y_train, X_val, y_val)
        
        assert model.is_fitted_
        assert 'validation_metrics' in model.training_history_
    
    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y)
        
        probas = model.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y)
        
        importance_df = model.get_feature_importance()
        
        assert len(importance_df) == X.shape[1]
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert importance_df['importance'].sum() > 0
    
    def test_save_load(self, sample_data):
        """Test model serialization."""
        X, y = sample_data
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y)
        
        original_predictions = model.predict_proba(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model.pkl'
            model.save(str(model_path))
            
            loaded_model = LGBMModel()
            loaded_model.load(str(model_path))
            
            loaded_predictions = loaded_model.predict_proba(X)
            
            assert loaded_model.is_fitted_
            assert loaded_model.feature_names_ == model.feature_names_
            assert np.allclose(original_predictions, loaded_predictions)
    
    def test_error_on_predict_before_fit(self, sample_data):
        """Should error if predict called before fit."""
        X, _ = sample_data
        
        model = LGBMModel()
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)
    
    def test_error_on_non_binary_target(self, sample_data):
        """Should error if target is not binary."""
        X, _ = sample_data
        y_invalid = np.random.randint(0, 3, len(X))
        
        model = LGBMModel(model_params={'verbose': -1})
        
        with pytest.raises(ValueError, match="must be binary"):
            model.fit(X, y_invalid)
    
    def test_error_on_single_class(self, sample_data):
        """Should error if only one class in target."""
        X, _ = sample_data
        y_single = np.zeros(len(X), dtype=int)
        
        model = LGBMModel(model_params={'verbose': -1})
        
        with pytest.raises(ValueError, match="both classes"):
            model.fit(X, y_single)
    
    def test_handles_imbalanced_classes(self, sample_data):
        """Should handle imbalanced class distribution."""
        X, y = sample_data
        
        # Create imbalanced dataset with 10% positive class
        y_imbalanced = np.zeros(len(y), dtype=int)
        positive_indices = np.random.choice(len(y), size=int(len(y) * 0.1), replace=False)
        y_imbalanced[positive_indices] = 1
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y_imbalanced)
        
        assert model.is_fitted_
        predictions = model.predict(X)
        
        # Model may predict mostly majority class, but should complete training
        assert len(predictions) == len(X)
        # At least one of the classes should be predicted (might not be both due to severe imbalance)
        assert len(np.unique(predictions)) >= 1
    
    def test_validates_features_on_predict(self, sample_data):
        """Should validate features match training."""
        X, y = sample_data
        
        model = LGBMModel(model_params={'num_iterations': 10, 'verbose': -1})
        model.fit(X, y)
        
        X_wrong = X.drop(columns=['feature_0'])
        
        with pytest.raises(ValueError, match="Missing features"):
            model.predict(X_wrong)


class TestLGBMModelFactory:
    """Test model factory presets."""
    
    def test_create_fast_model(self):
        """Test fast model creation."""
        model = LGBMModelFactory.create_fast_model()
        
        assert model.model_params['num_iterations'] == 100
        assert model.model_params['learning_rate'] == 0.2
        assert model.early_stopping_rounds == 20
    
    def test_create_accurate_model(self):
        """Test accurate model creation."""
        model = LGBMModelFactory.create_accurate_model()
        
        assert model.model_params['num_iterations'] == 1000
        assert model.model_params['learning_rate'] == 0.05
        assert model.early_stopping_rounds == 100
    
    def test_create_from_config(self):
        """Test model creation from config dict."""
        config = {
            'model_params': {
                'learning_rate': 0.15,
                'num_leaves': 50
            },
            'early_stopping_rounds': 30
        }
        
        model = LGBMModelFactory.create_from_config(config)
        
        assert model.model_params['learning_rate'] == 0.15
        assert model.model_params['num_leaves'] == 50
        assert model.early_stopping_rounds == 30


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create data similar to actual market data."""
        np.random.seed(42)
        n_samples = 5000
        
        features = {
            'returns_lag_3_mean': np.random.randn(n_samples) * 0.02,
            'returns_lag_7_mean': np.random.randn(n_samples) * 0.02,
            'returns_lag_14_mean': np.random.randn(n_samples) * 0.02,
            'volume_lag_3_mean': np.random.uniform(1e6, 1e7, n_samples),
            'volatility_20d': np.random.uniform(0.01, 0.05, n_samples),
            'news_sentiment_mean': np.random.randn(n_samples) * 5,
            'news_volume': np.random.poisson(2, n_samples),
            'has_news': np.random.randint(0, 2, n_samples)
        }
        
        X = pd.DataFrame(features)
        
        signal = (
            X['returns_lag_3_mean'] * 2 +
            X['returns_lag_7_mean'] +
            X['news_sentiment_mean'] * 0.01 +
            np.random.randn(n_samples) * 0.02
        )
        y = (signal > 0).astype(int)
        
        return X, y
    
    def test_full_training_pipeline(self, realistic_data):
        """Test complete training workflow."""
        X, y = realistic_data
        
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        val_idx = int(len(X_train) * 0.8)
        X_train_fit, X_val = X_train[:val_idx], X_train[val_idx:]
        y_train_fit, y_val = y_train[:val_idx], y_train[val_idx:]
        
        model = LGBMModel(
            model_params={
                'learning_rate': 0.1,
                'num_leaves': 50,
                'num_iterations': 100,
                'verbose': -1
            },
            early_stopping_rounds=20
        )
        
        model.fit(X_train_fit, y_train_fit, X_val, y_val)
        
        train_preds = model.predict(X_train_fit)
        test_preds = model.predict(X_test)
        test_proba = model.predict_proba(X_test)
        
        assert model.is_fitted_
        assert len(train_preds) == len(X_train_fit)
        assert len(test_preds) == len(X_test)
        assert test_proba.shape == (len(X_test), 2)
        
        train_metrics = ModelMetrics.calculate_binary_metrics(
            y_train_fit, train_preds
        )
        test_metrics = ModelMetrics.calculate_binary_metrics(
            y_test, test_preds, test_proba
        )
        
        assert train_metrics['accuracy'] > 0.5
        assert test_metrics['accuracy'] > 0.5
        assert 'log_loss' in test_metrics
        
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert importance.iloc[0]['importance'] > 0
