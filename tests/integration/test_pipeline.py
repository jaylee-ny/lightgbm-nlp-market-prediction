import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.config.settings import Config, DataConfig, ModelConfig, BacktestConfig
from src.pipeline.main_pipeline import TradingPipeline


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
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
        
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
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
    
    @pytest.fixture
    def fast_config(self):
        """Create fast configuration for testing."""
        return Config(
            data=DataConfig(
                train_start_date="2023-01-01",
                train_end_date="2023-06-30",
                test_start_date="2023-07-01",
                test_end_date="2023-12-31"
            ),
            model=ModelConfig(
                learning_rate=0.1,
                num_leaves=31,
                num_iterations=10,
                verbose=-1
            ),
            backtest=BacktestConfig(
                train_window_years=0.25,
                test_window_months=1
            ),
            output_dir="outputs/test"
        )
    
    def test_pipeline_without_news(self, sample_market_data, fast_config):
        """Test pipeline without news data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_config.output_dir = tmpdir
            
            pipeline = TradingPipeline(fast_config)
            results = pipeline.run(sample_market_data, news_df=None)
            
            assert 'backtest_results' in results
            assert 'performance_metrics' in results
            assert len(results['backtest_results']['fold_results']) > 0
    
    def test_pipeline_with_news(
        self,
        sample_market_data,
        sample_news_data,
        fast_config
    ):
        """Test pipeline with news data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_config.output_dir = tmpdir
            fast_config.features.use_news = True
            
            pipeline = TradingPipeline(fast_config)
            results = pipeline.run(sample_market_data, sample_news_data)
            
            assert 'backtest_results' in results
            assert 'performance_metrics' in results
    
    def test_pipeline_creates_outputs(
        self,
        sample_market_data,
        fast_config
    ):
        """Test that pipeline creates output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_config.output_dir = tmpdir
            
            pipeline = TradingPipeline(fast_config)
            pipeline.run(sample_market_data)
            
            output_dir = Path(tmpdir)
            
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'predictions.csv').exists()
            assert (output_dir / 'feature_importance.csv').exists()
            assert (output_dir / 'performance_report.txt').exists()
    
    def test_pipeline_validation_errors(self, fast_config):
        """Test pipeline handles validation errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_config.output_dir = tmpdir
            
            # Missing required columns
            bad_data = pd.DataFrame({
                'time': pd.date_range('2023-01-01', periods=100),
                'assetCode': ['AAPL'] * 100
            })
            
            pipeline = TradingPipeline(fast_config)
            
            with pytest.raises(ValueError):
                pipeline.run(bad_data)
    
    def test_pipeline_performance_metrics(
        self,
        sample_market_data,
        fast_config
    ):
        """Test that performance metrics are calculated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_config.output_dir = tmpdir
            
            pipeline = TradingPipeline(fast_config)
            results = pipeline.run(sample_market_data)
            
            metrics = results['performance_metrics']['metrics']
            
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'win_rate' in metrics
            assert 'total_return' in metrics
    
    def test_pipeline_feature_importance(
        self,
        sample_market_data,
        fast_config
    ):
        """Test that feature importance is calculated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_config.output_dir = tmpdir
            
            pipeline = TradingPipeline(fast_config)
            results = pipeline.run(sample_market_data)
            
            importance = results['backtest_results']['feature_importance']
            
            assert len(importance) > 0
            assert 'feature' in importance.columns
            assert 'mean_importance' in importance.columns


class TestEndToEnd:
    """End-to-end tests with realistic scenarios."""
    
    def test_complete_workflow(self):
        """Test complete workflow from config to results."""
        np.random.seed(42)
        
        # Create realistic data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        market_data = []
        for date in dates:
            for asset in assets:
                signal = np.random.randn() * 0.02
                market_data.append({
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
        
        market_df = pd.DataFrame(market_data)
        
        # Create configuration
        config = Config(
            data=DataConfig(
                train_start_date="2023-01-01",
                train_end_date="2023-09-30",
                test_start_date="2023-10-01",
                test_end_date="2023-12-31"
            ),
            model=ModelConfig(
                learning_rate=0.1,
                num_leaves=50,
                num_iterations=50
            ),
            backtest=BacktestConfig(
                train_window_years=0.5,
                test_window_months=2
            )
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            
            # Run pipeline
            pipeline = TradingPipeline(config)
            results = pipeline.run(market_df)
            
            # Verify results structure
            assert 'backtest_results' in results
            assert 'performance_metrics' in results
            
            # Verify backtest ran
            fold_results = results['backtest_results']['fold_results']
            assert len(fold_results) > 0
            
            # Verify metrics calculated
            metrics = results['performance_metrics']['metrics']
            assert 'sharpe_ratio' in metrics
            assert isinstance(metrics['sharpe_ratio'], (int, float))
            
            # Verify outputs created
            output_dir = Path(tmpdir)
            assert (output_dir / 'predictions.csv').exists()
            assert (output_dir / 'performance_report.txt').exists()
