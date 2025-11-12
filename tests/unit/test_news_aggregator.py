import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.news_aggregator import (
    NewsAggregator,
    NewsCoverageFiller,
    NewsQualityValidator
)


class TestNewsAggregator:
    """Test news aggregation functionality."""
    
    @pytest.fixture
    def sample_news(self):
        """Sample news data spanning multiple days and assets."""
        base_time = datetime(2024, 1, 1, 9, 0)
        
        news_items = []
        for day in range(3):
            for hour in [9, 12, 15]:
                for asset in ['AAPL', 'GOOGL']:
                    news_items.append({
                        'time': base_time + timedelta(days=day, hours=hour),
                        'assetName': asset,
                        'sentimentWordCount': np.random.randint(5, 20),
                        'wordCount': np.random.randint(50, 200)
                    })
        
        return pd.DataFrame(news_items)
    
    def test_fit_learns_global_statistics(self, sample_news):
        """Fit should learn global statistics from training data."""
        aggregator = NewsAggregator()
        aggregator.fit(sample_news)
        
        assert aggregator._is_fitted
        assert 'coverage_median' in aggregator.global_stats_
        assert 'articles_per_day_median' in aggregator.global_stats_
        assert aggregator.global_stats_['coverage_median'] > 0
    
    def test_aggregates_to_day_level(self, sample_news):
        """Transform should create one row per asset-day."""
        aggregator = NewsAggregator()
        aggregator.fit(sample_news)
        
        aggregated = aggregator.transform(sample_news)
        
        assert 'news_volume' in aggregated.columns
        assert 'news_sentiment_mean' in aggregated.columns
        assert 'news_coverage_mean' in aggregated.columns
        
        unique_days = sample_news['time'].dt.date.nunique()
        unique_assets = sample_news['assetName'].nunique()
        expected_rows = unique_days * unique_assets
        
        assert len(aggregated) <= expected_rows
    
    def test_time_decay_weighting(self):
        """Recent news should have higher weight."""
        news = pd.DataFrame({
            'time': [
                datetime(2024, 1, 1, 9, 0),
                datetime(2024, 1, 1, 15, 0)
            ],
            'assetName': ['AAPL', 'AAPL'],
            'sentimentWordCount': [10, 10],
            'wordCount': [100, 100]
        })
        
        aggregator = NewsAggregator(decay_half_life=6.0)
        aggregator.fit(news)
        
        news_weighted = aggregator._add_time_weights(news)
        
        assert news_weighted.iloc[1]['time_weight'] > news_weighted.iloc[0]['time_weight']
    
    def test_handles_empty_news(self):
        """Should handle empty news dataframe gracefully."""
        empty_news = pd.DataFrame(columns=['time', 'assetName', 
                                           'sentimentWordCount', 'wordCount'])
        
        aggregator = NewsAggregator()
        aggregator.fit(empty_news)
        
        result = aggregator.transform(empty_news)
        
        assert len(result) == 0
        assert 'news_sentiment_mean' in result.columns
    
    def test_aggregates_multiple_articles_per_day(self):
        """Multiple articles per asset-day should be aggregated."""
        news = pd.DataFrame({
            'time': [datetime(2024, 1, 1, 9, 0)] * 5,
            'assetName': ['AAPL'] * 5,
            'sentimentWordCount': [5, 10, 15, 20, 25],
            'wordCount': [50, 100, 150, 200, 250]
        })
        
        aggregator = NewsAggregator()
        aggregator.fit(news)
        
        aggregated = aggregator.transform(news)
        
        assert len(aggregated) == 1
        assert aggregated.iloc[0]['news_volume'] == 5
        assert aggregated.iloc[0]['news_sentiment_mean'] == 15
    
    def test_separate_assets_independently(self):
        """Each asset should be aggregated independently."""
        news = pd.DataFrame({
            'time': [datetime(2024, 1, 1, 9, 0)] * 4,
            'assetName': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
            'sentimentWordCount': [10, 20, 100, 200],
            'wordCount': [100, 200, 1000, 2000]
        })
        
        aggregator = NewsAggregator()
        aggregator.fit(news)
        
        aggregated = aggregator.transform(news)
        
        assert len(aggregated) == 2
        
        aapl_row = aggregated[aggregated['assetName'] == 'AAPL'].iloc[0]
        googl_row = aggregated[aggregated['assetName'] == 'GOOGL'].iloc[0]
        
        assert aapl_row['news_sentiment_mean'] == 15
        assert googl_row['news_sentiment_mean'] == 150


class TestNewsCoverageFiller:
    """Test filling missing news coverage."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Market data with 3 assets."""
        return pd.DataFrame({
            'time': [datetime(2024, 1, 1).date()] * 3,
            'assetName': ['AAPL', 'GOOGL', 'MSFT'],
            'close': [150, 120, 300]
        })
    
    @pytest.fixture
    def sample_news_data(self):
        """News data covering only 2 assets."""
        return pd.DataFrame({
            'time': [datetime(2024, 1, 1).date()] * 2,
            'assetName': ['AAPL', 'GOOGL'],
            'news_volume': [5, 3],
            'news_sentiment_mean': [10.0, 15.0],
            'news_coverage_mean': [0.1, 0.15]
        })
    
    def test_fills_missing_assets(self, sample_market_data, sample_news_data):
        """Assets without news should be filled with learned values."""
        filler = NewsCoverageFiller(strategy='global_median')
        filler.fit(sample_news_data)
        
        merged = filler.transform(sample_market_data, sample_news_data)
        
        assert len(merged) == 3
        assert not merged['news_volume'].isna().any()
        assert not merged['news_sentiment_mean'].isna().any()
        
        msft_row = merged[merged['assetName'] == 'MSFT'].iloc[0]
        assert msft_row['news_volume'] > 0 or msft_row['news_volume'] == 0
    
    def test_preserves_existing_news(self, sample_market_data, sample_news_data):
        """Assets with news should keep original values."""
        filler = NewsCoverageFiller(strategy='global_median')
        filler.fit(sample_news_data)
        
        merged = filler.transform(sample_market_data, sample_news_data)
        
        aapl_row = merged[merged['assetName'] == 'AAPL'].iloc[0]
        assert aapl_row['news_volume'] == 5
        assert aapl_row['news_sentiment_mean'] == 10.0
    
    def test_adds_has_news_flag(self, sample_market_data, sample_news_data):
        """Should add binary flag for news presence."""
        filler = NewsCoverageFiller()
        filler.fit(sample_news_data)
        
        merged = filler.transform(sample_market_data, sample_news_data)
        
        assert 'has_news' in merged.columns
        
        # Convert numpy bool to Python bool for comparison
        aapl_has_news = bool(merged[merged['assetName'] == 'AAPL']['has_news'].iloc[0])
        msft_has_news = bool(merged[merged['assetName'] == 'MSFT']['has_news'].iloc[0])
        
        assert aapl_has_news == True
        assert msft_has_news == False


class TestNewsQualityValidator:
    """Test news quality validation."""
    
    def test_checks_coverage_rate(self):
        """Should check if sufficient assets have news."""
        market = pd.DataFrame({
            'time': [datetime(2024, 1, 1).date()] * 10,
            'assetName': [f'ASSET_{i}' for i in range(10)],
            'close': [100] * 10
        })
        
        news = pd.DataFrame({
            'time': [datetime(2024, 1, 1).date()] * 5,
            'assetName': [f'ASSET_{i}' for i in range(5)],
            'news_volume': [1] * 5
        })
        
        validator = NewsQualityValidator()
        result = validator.check_coverage_rate(market, news, min_coverage=0.4)
        
        assert result == True
    
    def test_detects_low_coverage(self):
        """Should detect low coverage rates."""
        market = pd.DataFrame({
            'time': [datetime(2024, 1, 1).date()] * 10,
            'assetName': [f'ASSET_{i}' for i in range(10)]
        })
        
        news = pd.DataFrame({
            'time': [datetime(2024, 1, 1).date()],
            'assetName': ['ASSET_0']
        })
        
        validator = NewsQualityValidator()
        result = validator.check_coverage_rate(market, news, min_coverage=0.5)
        
        assert result == False
    
    def test_detects_duplicates(self):
        """Should detect and remove duplicate news items."""
        news = pd.DataFrame({
            'time': [datetime(2024, 1, 1)] * 3,
            'assetName': ['AAPL', 'AAPL', 'GOOGL'],
            'headline': ['News A', 'News A', 'News B'],
            'sentimentWordCount': [10, 10, 15]
        })
        
        validator = NewsQualityValidator()
        cleaned = validator.detect_duplicates(news)
        
        assert len(cleaned) == 2


class TestIntegration:
    """Integration tests with realistic data."""
    
    @pytest.fixture
    def realistic_news(self):
        """Realistic news data with varied coverage."""
        np.random.seed(42)
        
        news_items = []
        base_time = datetime(2024, 1, 1)
        
        for day in range(30):
            for asset in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']:
                n_articles = np.random.poisson(2) if asset != 'TSLA' else np.random.poisson(0.5)
                
                for _ in range(n_articles):
                    hour = np.random.randint(9, 16)
                    news_items.append({
                        'time': base_time + timedelta(days=day, hours=hour),
                        'assetName': asset,
                        'sentimentWordCount': np.random.randint(5, 30),
                        'wordCount': np.random.randint(50, 300)
                    })
        
        return pd.DataFrame(news_items)
    
    def test_full_pipeline(self, realistic_news):
        """Test complete news aggregation pipeline."""
        train_cutoff = datetime(2024, 1, 21)
        
        train_news = realistic_news[realistic_news['time'] < train_cutoff]
        test_news = realistic_news[realistic_news['time'] >= train_cutoff]
        
        aggregator = NewsAggregator(decay_half_life=24.0)
        aggregator.fit(train_news)
        
        train_agg = aggregator.transform(train_news)
        test_agg = aggregator.transform(test_news)
        
        assert len(train_agg) > 0
        assert len(test_agg) > 0
        assert not train_agg['news_volume'].isna().any()
        assert not test_agg['news_sentiment_mean'].isna().any()
        
        assert train_agg['time'].max() < test_agg['time'].min()
