import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.engine import BacktestEngine, BacktestResult
from src.utils.time_utils import TimeSeriesSplitter


class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_backtest_result_creation(self):
        """Test creating a BacktestResult."""
        result = BacktestResult(
            fold=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 12, 31),
            train_size=1000,
            test_size=200,
            predictions=np.array([0.1, 0.9, 0.3]),
            actual=np.array([0, 1, 0]),
            metrics={'accuracy': 0.75},
            feature_importance=pd.DataFrame({'feature': ['f1'], 'importance': [0.5]})
        )
        
        assert result.fold == 1
        assert result.train_size == 1000
        assert result.test_size == 200
        assert len(result.predictions) == 3
        assert 'accuracy' in result.metrics


class TestBacktestEngine:
    """Test backtesting engine functionality."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for backtesting."""
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        data = []
        for date in dates:
            for asset in assets:
                data.append({
                    'time': date,
                    'assetCode': asset,
                    'assetName': asset,
                    'returnsClosePrevRaw1': np.random.randn() * 0.02,
                    'returnsOpenPrevRaw1': np.random.randn() * 0.02,
                    'returnsClosePrevMktres1': np.random.randn() * 0.015,
                    'returnsOpenPrevMktres1': np.random.randn() * 0.015,
                    'volume': np.random.uniform(1e6, 1e7),
                    'close': np.random.uniform(100, 200),
                    'returnsOpenNextMktres10': np.random.randn() * 0.03
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data."""
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        data = []
        for date in dates:
            for asset in assets:
                if np.random.random() > 0.3:
                    data.append({
                        'time': date,
                        'assetName': asset,
                        'sentimentWordCount': np.random.randint(5, 20),
                        'wordCount': np.random.randint(50, 200)
                    })
        
        return pd.DataFrame(data)
    
    def test_engine_initialization(self):
        """Test engine can be initialized."""
        splitter = TimeSeriesSplitter(
            train_window_years=1,
            test_window_months=3
        )
        
        engine = BacktestEngine(time_splitter=splitter)
        
        assert engine.time_splitter is not None
        assert engine.results_ == []
    
    def test_run_backtest_without_news(self, sample_market_data):
        """Test running backtest without news data."""
        splitter = TimeSeriesSplitter(
            train_window_years=0.5,
            test_window_months=2
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            lag_feature_config={'lag_windows': [3, 7]},
            model_params={
                'learning_rate': 0.1,
                'num_leaves': 31,
                'num_iterations': 10,
                'verbose': -1
            }
        )
        
        results = engine.run(sample_market_data)
        
        assert len(results) > 0
        assert all(isinstance(r, BacktestResult) for r in results)
        assert all(r.test_size > 0 for r in results)
    
    def test_run_backtest_with_news(self, sample_market_data, sample_news_data):
        """Test running backtest with news data."""
        splitter = TimeSeriesSplitter(
            train_window_years=0.5,
            test_window_months=2
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            lag_feature_config={'lag_windows': [3, 7]},
            model_params={
                'learning_rate': 0.1,
                'num_leaves': 31,
                'num_iterations': 10,
                'verbose': -1
            }
        )
        
        results = engine.run(sample_market_data, sample_news_data)
        
        assert len(results) > 0
    
    def test_temporal_separation(self, sample_market_data):
        """Test that train and test periods don't overlap."""
        splitter = TimeSeriesSplitter(
            train_window_years=0.5,
            test_window_months=2
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            model_params={'num_iterations': 10, 'verbose': -1}
        )
        
        results = engine.run(sample_market_data)
        
        for result in results:
            assert result.train_end < result.test_start
    
    def test_aggregated_metrics(self, sample_market_data):
        """Test aggregated metrics calculation."""
        splitter = TimeSeriesSplitter(
            train_window_years=0.5,
            test_window_months=2
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            model_params={'num_iterations': 10, 'verbose': -1}
        )
        
        engine.run(sample_market_data)
        metrics = engine.get_aggregated_metrics()
        
        assert 'accuracy' in metrics
        assert 'mean' in metrics['accuracy']
        assert 'std' in metrics['accuracy']
    
    def test_get_all_predictions(self, sample_market_data):
        """Test getting all predictions."""
        splitter = TimeSeriesSplitter(
            train_window_years=0.5,
            test_window_months=2
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            model_params={'num_iterations': 10, 'verbose': -1}
        )
        
        engine.run(sample_market_data)
        predictions_df = engine.get_all_predictions()
        
        assert len(predictions_df) > 0
        assert 'prediction' in predictions_df.columns
        assert 'actual' in predictions_df.columns
        assert 'fold' in predictions_df.columns
    
    def test_feature_importance_summary(self, sample_market_data):
        """Test feature importance aggregation."""
        splitter = TimeSeriesSplitter(
            train_window_years=0.5,
            test_window_months=2
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            lag_feature_config={'lag_windows': [3]},
            model_params={'num_iterations': 10, 'verbose': -1}
        )
        
        engine.run(sample_market_data)
        importance = engine.get_feature_importance_summary()
        
        assert len(importance) > 0
        assert 'feature' in importance.columns
        assert 'mean_importance' in importance.columns
        assert 'std_importance' in importance.columns


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic market data."""
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        data = []
        for date in dates:
            for asset in assets:
                signal = np.random.randn() * 0.02
                data.append({
                    'time': date,
                    'assetCode': asset,
                    'assetName': asset,
                    'returnsClosePrevRaw1': signal + np.random.randn() * 0.005,
                    'returnsOpenPrevRaw1': signal + np.random.randn() * 0.005,
                    'returnsClosePrevMktres1': signal,
                    'returnsOpenPrevMktres1': signal,
                    'volume': np.random.uniform(1e6, 1e7),
                    'close': 150 + np.random.randn() * 10,
                    'returnsOpenNextMktres10': signal * 0.5 + np.random.randn() * 0.02
                })
        
        return pd.DataFrame(data)
    
    def test_full_backtest_pipeline(self, realistic_data):
        """Test complete backtesting workflow."""
        splitter = TimeSeriesSplitter(
            train_window_years=1,
            test_window_months=3
        )
        
        engine = BacktestEngine(
            time_splitter=splitter,
            lag_feature_config={
                'lag_windows': [3, 7, 14],
                'feature_columns': [
                    'returnsClosePrevRaw1',
                    'returnsOpenPrevRaw1',
                    'volume'
                ]
            },
            model_params={
                'learning_rate': 0.1,
                'num_leaves': 50,
                'num_iterations': 50,
                'verbose': -1
            }
        )
        
        results = engine.run(realistic_data)
        
        assert len(results) >= 1
        
        for result in results:
            assert result.metrics['accuracy'] > 0.4
            assert result.metrics['accuracy'] < 1.0
            assert 0 <= result.metrics.get('log_loss', 1.0) <= 2.0
        
        metrics = engine.get_aggregated_metrics()
        assert 'accuracy' in metrics
        assert 0.4 <= metrics['accuracy']['mean'] <= 0.7
        
        predictions = engine.get_all_predictions()
        assert len(predictions) > 0
        
        importance = engine.get_feature_importance_summary()
        assert len(importance) > 0
