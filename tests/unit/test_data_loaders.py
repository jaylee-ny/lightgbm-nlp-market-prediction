"""Tests for data loaders."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.data.loaders import DataLoader
from src.data.validators import DataValidator
from src.config import DataConfig


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_market_csv(temp_data_dir):
    """Create sample market data CSV."""
    csv_path = temp_data_dir / "marketdata_sample.csv"
    
    df = pd.DataFrame({
        'time': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'assetCode': ['AAPL.O', 'GOOGL.O', 'MSFT.O'],
        'assetName': ['Apple Inc', 'Alphabet Inc', 'Microsoft Corp'],
        'open': [100.0, 200.0, 150.0],
        'close': [105.0, 195.0, 155.0],
        'volume': [1000000, 2000000, 1500000],
        'returnsClosePrevRaw1': [0.01, -0.02, 0.03],
        'returnsOpenPrevRaw1': [0.01, -0.02, 0.03],
        'returnsClosePrevMktres1': [0.01, -0.02, 0.03],
        'returnsOpenPrevMktres1': [0.01, -0.02, 0.03],
        'returnsClosePrevRaw10': [0.05, -0.03, 0.04],
        'returnsOpenPrevRaw10': [0.05, -0.03, 0.04],
        'returnsClosePrevMktres10': [0.05, -0.03, 0.04],
        'returnsOpenPrevMktres10': [0.05, -0.03, 0.04],
        'returnsOpenNextMktres10': [0.01, -0.02, 0.03]
    })
    
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_news_csv(temp_data_dir):
    """Create sample news data CSV."""
    csv_path = temp_data_dir / "news_sample.csv"
    
    df = pd.DataFrame({
        'time': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'assetName': ['Apple Inc', 'Alphabet Inc', 'Microsoft Corp'],
        'sentimentWordCount': [10, 15, 5],
        'wordCount': [100, 200, 50]
    })
    
    df.to_csv(csv_path, index=False)
    return csv_path


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def test_load_market_data_success(self, temp_data_dir, sample_market_csv):
        """Test successful market data loading."""
        config = DataConfig(raw_data_dir=temp_data_dir)
        loader = DataLoader(config)
        
        df = loader.load_market_data()
        
        assert len(df) == 3
        assert 'time' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['time'])
        assert df['assetCode'].nunique() == 3
    
    def test_load_market_data_file_not_found(self, temp_data_dir):
        """Test error when file doesn't exist."""
        config = DataConfig(raw_data_dir=temp_data_dir)
        loader = DataLoader(config)
        
        with pytest.raises(FileNotFoundError, match="Market data not found"):
            loader.load_market_data()
    
    def test_load_market_data_missing_columns(self, temp_data_dir):
        """Test error when required columns are missing."""
        csv_path = temp_data_dir / "marketdata_sample.csv"
        df = pd.DataFrame({
            'time': ['2024-01-01'],
            'assetCode': ['AAPL.O']
        })
        df.to_csv(csv_path, index=False)
        
        config = DataConfig(raw_data_dir=temp_data_dir)
        loader = DataLoader(config)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_market_data()
    
    def test_datetime_parsing(self, temp_data_dir, sample_market_csv):
        """Test that datetime is correctly parsed."""
        config = DataConfig(raw_data_dir=temp_data_dir)
        loader = DataLoader(config)
        
        df = loader.load_market_data()
        
        assert df['time'].dtype == 'datetime64[ns]'
        assert df['time'].min() == pd.Timestamp('2024-01-01')
        assert df['time'].max() == pd.Timestamp('2024-01-03')
    
    def test_load_news_data_success(self, temp_data_dir, sample_news_csv):
        """Test successful news data loading."""
        config = DataConfig(raw_data_dir=temp_data_dir)
        loader = DataLoader(config)
        
        df = loader.load_news_data()
        
        assert len(df) == 3
        assert 'time' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['time'])
    
    def test_load_both_datasets(self, temp_data_dir, sample_market_csv, sample_news_csv):
        """Test loading both market and news data."""
        config = DataConfig(raw_data_dir=temp_data_dir)
        loader = DataLoader(config)
        
        market_df, news_df = loader.load_data()
        
        assert len(market_df) == 3
        assert len(news_df) == 3
        assert 'assetCode' in market_df.columns
        assert 'assetName' in news_df.columns


class TestDataValidator:
    """Test DataValidator functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10),
            'assetCode': ['AAPL'] * 10,
            'close': [100 + i for i in range(10)],
            'volume': [1000000 + i * 1000 for i in range(10)]
        })
    
    def test_validate_no_missing_columns_success(self, sample_data):
        """Test validation passes with all required columns."""
        validator = DataValidator()
        required = ['time', 'assetCode', 'close', 'volume']
        
        assert validator.validate_no_missing_required_columns(sample_data, required)
    
    def test_validate_no_missing_columns_failure(self, sample_data):
        """Test validation fails when columns are missing."""
        validator = DataValidator()
        required = ['time', 'assetCode', 'close', 'volume', 'missing_column']
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validator.validate_no_missing_required_columns(sample_data, required)
    
    def test_validate_price_sanity_success(self, sample_data):
        """Test validation passes with valid prices."""
        validator = DataValidator()
        assert validator.validate_price_sanity(sample_data)
    
    def test_validate_price_sanity_negative_prices(self, sample_data):
        """Test validation fails with negative prices."""
        sample_data.loc[0, 'close'] = -100
        validator = DataValidator()
        
        with pytest.raises(ValueError, match="negative closing prices"):
            validator.validate_price_sanity(sample_data)
    
    def test_validate_temporal_ordering_success(self, sample_data):
        """Test validation passes with chronological data."""
        validator = DataValidator()
        assert validator.validate_temporal_ordering(sample_data)
    
    def test_validate_temporal_ordering_failure(self, sample_data):
        """Test validation fails with non-chronological data."""
        sample_data = sample_data.sort_values('time', ascending=False)
        validator = DataValidator()
        
        with pytest.raises(ValueError, match="not chronological"):
            validator.validate_temporal_ordering(sample_data)
    
    def test_validate_no_duplicate_timestamps(self, sample_data):
        """Test validation passes with no duplicates."""
        validator = DataValidator()
        assert validator.validate_no_duplicate_timestamps(sample_data)
    
    def test_validate_duplicate_timestamps_failure(self):
        """Test validation fails with duplicate timestamps."""
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5).tolist() * 2,
            'assetCode': ['AAPL'] * 5 + ['GOOGL'] * 5
        })
        df = pd.concat([df, df.iloc[[0]]])
        
        validator = DataValidator()
        
        with pytest.raises(ValueError, match="duplicate entries"):
            validator.validate_no_duplicate_timestamps(df)
    
    def test_generate_data_quality_report(self, sample_data):
        """Test data quality report generation."""
        validator = DataValidator()
        report = validator.generate_data_quality_report(sample_data)
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'date_range' in report
        assert report['total_rows'] == 10
