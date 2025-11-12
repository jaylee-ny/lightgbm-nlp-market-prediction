import pytest
import pandas as pd
import numpy as np

from src.models.optimizer import BayesianOptimizer, GridSearchOptimizer


class TestBayesianOptimizer:
    """Test Bayesian optimization functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for optimization."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        return X, y
    
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        optimizer = BayesianOptimizer(n_trials=10, cv_folds=2)
        
        assert optimizer.n_trials == 10
        assert optimizer.cv_folds == 2
        assert optimizer.best_params_ is None
    
    def test_optimize_with_default_space(self, sample_data):
        """Test optimization with default parameter space."""
        X, y = sample_data
        
        optimizer = BayesianOptimizer(
            n_trials=5,
            cv_folds=2,
            random_state=42
        )
        
        best_params = optimizer.optimize(X, y)
        
        assert optimizer.best_params_ is not None
        assert optimizer.best_score_ is not None
        assert 'learning_rate' in best_params
        assert 'num_leaves' in best_params
        assert len(optimizer.optimization_history_) == 5
    
    def test_optimize_with_custom_space(self, sample_data):
        """Test optimization with custom parameter space."""
        X, y = sample_data
        
        custom_space = {
            'learning_rate': {'type': 'float', 'low': 0.05, 'high': 0.2},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 100}
        }
        
        optimizer = BayesianOptimizer(n_trials=3, cv_folds=2)
        best_params = optimizer.optimize(X, y, param_space=custom_space)
        
        assert 0.05 <= best_params['learning_rate'] <= 0.2
        assert 20 <= best_params['num_leaves'] <= 100
    
    def test_optimization_history(self, sample_data):
        """Test optimization history tracking."""
        X, y = sample_data
        
        optimizer = BayesianOptimizer(n_trials=5, cv_folds=2)
        optimizer.optimize(X, y)
        
        history_df = optimizer.get_optimization_history()
        
        assert len(history_df) == 5
        assert 'trial' in history_df.columns
        assert 'mean_score' in history_df.columns
        assert 'std_score' in history_df.columns
        assert 'learning_rate' in history_df.columns
    
    def test_categorical_parameter(self, sample_data):
        """Test optimization with categorical parameters."""
        X, y = sample_data
        
        custom_space = {
            'learning_rate': {'type': 'float', 'low': 0.05, 'high': 0.2},
            'boosting_type': {
                'type': 'categorical',
                'choices': ['gbdt', 'dart']
            }
        }
        
        optimizer = BayesianOptimizer(n_trials=3, cv_folds=2)
        best_params = optimizer.optimize(X, y, param_space=custom_space)
        
        assert best_params['boosting_type'] in ['gbdt', 'dart']
    
    def test_walk_forward_cv(self, sample_data):
        """Test that walk-forward CV is used correctly."""
        X, y = sample_data
        
        optimizer = BayesianOptimizer(n_trials=2, cv_folds=3)
        optimizer.optimize(X, y)
        
        for entry in optimizer.optimization_history_:
            assert len(entry['cv_scores']) == 3


class TestGridSearchOptimizer:
    """Test grid search functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        n_samples = 500
        
        X = pd.DataFrame({
            'feature_0': np.random.randn(n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        })
        
        y = (X['feature_0'] + X['feature_1'] > 0).astype(int)
        
        return X, y
    
    def test_grid_search_basic(self, sample_data):
        """Test basic grid search."""
        X, y = sample_data
        
        param_grid = {
            'learning_rate': [0.1, 0.2],
            'num_leaves': [20, 50]
        }
        
        optimizer = GridSearchOptimizer(cv_folds=2)
        best_params = optimizer.search(X, y, param_grid)
        
        assert best_params is not None
        assert best_params['learning_rate'] in [0.1, 0.2]
        assert best_params['num_leaves'] in [20, 50]
        assert len(optimizer.results_) == 4
    
    def test_grid_search_finds_best(self, sample_data):
        """Test that grid search finds best parameters."""
        X, y = sample_data
        
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31]
        }
        
        optimizer = GridSearchOptimizer(cv_folds=2)
        optimizer.search(X, y, param_grid)
        
        scores = [r['mean_score'] for r in optimizer.results_]
        best_idx = np.argmin(scores)
        
        assert optimizer.best_score_ == scores[best_idx]


class TestComparison:
    """Compare Bayesian optimization vs grid search."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate comparison data."""
        np.random.seed(42)
        n_samples = 800
        
        X = pd.DataFrame(
            np.random.randn(n_samples, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        y = (X['feature_0'] + X['feature_1'] * 0.5 > 0).astype(int)
        
        return X, y
    
    def test_bayesian_vs_grid_search(self, sample_data):
        """Compare Bayesian optimization efficiency."""
        X, y = sample_data
        
        grid_params = {
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50]
        }
        
        grid_optimizer = GridSearchOptimizer(cv_folds=2)
        grid_best = grid_optimizer.search(X, y, grid_params)
        grid_score = grid_optimizer.best_score_
        
        bayesian_space = {
            'learning_rate': {'type': 'float', 'low': 0.05, 'high': 0.2},
            'num_leaves': {'type': 'int', 'low': 31, 'high': 50}
        }
        
        bayesian_optimizer = BayesianOptimizer(
            n_trials=6,
            cv_folds=2,
            random_state=42
        )
        bayesian_best = bayesian_optimizer.optimize(X, y, param_space=bayesian_space)
        bayesian_score = bayesian_optimizer.best_score_
        
        assert grid_best is not None
        assert bayesian_best is not None
        
        # Bayesian should be competitive with same number of evaluations
        assert bayesian_score <= grid_score * 1.1


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic market-like data."""
        np.random.seed(42)
        n_samples = 2000
        
        features = {
            'returns_lag_3_mean': np.random.randn(n_samples) * 0.02,
            'returns_lag_7_mean': np.random.randn(n_samples) * 0.02,
            'volume_lag_3_mean': np.random.uniform(1e6, 1e7, n_samples),
            'news_sentiment': np.random.randn(n_samples) * 5
        }
        
        X = pd.DataFrame(features)
        
        signal = (
            X['returns_lag_3_mean'] * 2 +
            X['returns_lag_7_mean'] +
            X['news_sentiment'] * 0.005 +
            np.random.randn(n_samples) * 0.01
        )
        y = (signal > 0).astype(int)
        
        return X, y
    
    def test_full_optimization_pipeline(self, realistic_data):
        """Test complete optimization workflow."""
        X, y = realistic_data
        
        optimizer = BayesianOptimizer(
            n_trials=10,
            cv_folds=3,
            random_state=42
        )
        
        best_params = optimizer.optimize(
            X, y,
            objective_metric='log_loss',
            direction='minimize'
        )
        
        assert best_params is not None
        assert optimizer.best_score_ < 0.7
        
        history = optimizer.get_optimization_history()
        assert len(history) == 10
        
        assert history['mean_score'].min() == optimizer.best_score_
