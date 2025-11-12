import pytest
import pandas as pd
import numpy as np

from src.features.market_features import MarketFeatureEngineer


class TestMarketFeatureEngineer:
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'asset_code': ['AAPL'] * n,
            'time': pd.date_range('2024-01-01', periods=n),
            'open': np.random.uniform(100, 110, n),
            'high': np.random.uniform(110, 120, n),
            'low': np.random.uniform(90, 100, n),
            'close': np.random.uniform(100, 110, n),
            'volume': np.random.uniform(1e6, 1e7, n)
        })
    
    def test_create_features(self, sample_market_data):
        """Test feature creation."""
        engineer = MarketFeatureEngineer(window_sizes=[5, 10])
        result = engineer.create_features(sample_market_data)
        
        assert 'simple_return' in result.columns
        assert 'volatility_5d' in result.columns
        assert 'vwap_5d' in result.columns
    
    def test_returns_calculation(self, sample_market_data):
        """Test return metrics."""
        engineer = MarketFeatureEngineer()
        result = engineer.create_features(sample_market_data)
        
        assert 'simple_return' in result.columns
        assert 'log_return' in result.columns
        assert 'intraday_return' in result.columns
    
    def test_volatility_estimators(self, sample_market_data):
        """Test volatility calculations."""
        engineer = MarketFeatureEngineer(window_sizes=[20])
        result = engineer.create_features(sample_market_data)
        
        assert 'volatility_20d' in result.columns
        assert 'parkinson_volatility_20d' in result.columns
        assert 'garman_klass_volatility_20d' in result.columns
    
    def test_vwap_calculation(self, sample_market_data):
        """Test VWAP calculation."""
        engineer = MarketFeatureEngineer(window_sizes=[5])
        result = engineer.create_features(sample_market_data)
        
        assert 'vwap' in result.columns
        assert 'vwap_5d' in result.columns
        assert result['vwap'].notna().sum() > 0
    
    def test_handles_missing_data(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'asset_code': ['AAPL'] * 10,
            'close': [100, 102, np.nan, 104, 105, 106, np.nan, 108, 109, 110]
        })
        
        engineer = MarketFeatureEngineer()
        result = engineer.create_features(df)
        
        assert result is not None
