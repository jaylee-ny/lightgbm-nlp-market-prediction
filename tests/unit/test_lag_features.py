import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.lag_features import (
    LagFeatureGenerator,
    LeakageValidator,
    EdgeCaseHandler
)


class TestLagFeatureGenerator:
    """Test core lag feature functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Time series data for 2 assets over 30 days."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        data = []
        for asset in ['AAPL', 'GOOGL']:
            for date in dates:
                data.append({
                    'time': date,
                    'assetCode': asset,
                    'returnsClosePrevRaw1': np.random.randn() * 0.01,
                    'returnsOpenPrevRaw1': np.random.randn() * 0.01,
                    'returnsClosePrevMktres1': np.random.randn() * 0.01,
                    'returnsOpenPrevMktres1': np.random.randn() * 0.01,
                    'volume': np.random.uniform(1e6, 1e7),
                    'close': np.random.uniform(100, 200)
                })
        
        return pd.DataFrame(data)
    
    def test_fit_creates_statistics(self, sample_data):
        """Fit should learn statistics from training data."""
        generator = LagFeatureGenerator(
            lag_windows=[3, 7],
            feature_columns=['returnsClosePrevRaw1', 'volume']
        )
        
        generator.fit(sample_data)
        
        assert generator._is_fitted
        assert len(generator.statistics_) > 0
        assert len(generator.feature_names_) > 0
        
        for stats in generator.statistics_.values():
            assert 'mean' in stats
            assert 'median' in stats
            assert 'std' in stats
    
    def test_transform_adds_lag_features(self, sample_data):
        """Transform should add lag features to dataframe."""
        generator = LagFeatureGenerator(
            lag_windows=[3, 7],
            feature_columns=['returnsClosePrevRaw1']
        )
        
        train = sample_data[sample_data['time'] < '2024-01-21']
        test = sample_data[sample_data['time'] >= '2024-01-21']
        
        generator.fit(train)
        transformed = generator.transform(test)
        
        assert 'returnsClosePrevRaw1_lag_3_mean' in transformed.columns
        assert 'returnsClosePrevRaw1_lag_7_mean' in transformed.columns
        assert 'returnsClosePrevRaw1_lag_3_std' in transformed.columns
        assert len(transformed) == len(test)
    
    def test_no_look_ahead_bias(self, sample_data):
        """Lag features must not equal current values."""
        generator = LagFeatureGenerator(
            lag_windows=[3],
            feature_columns=['returnsClosePrevRaw1']
        )
        
        generator.fit(sample_data)
        transformed = generator.transform(sample_data)
        
        for asset in ['AAPL', 'GOOGL']:
            asset_df = transformed[
                transformed['assetCode'] == asset
            ].sort_values('time')
            
            for i in range(5, len(asset_df)):
                lag_val = asset_df.iloc[i]['returnsClosePrevRaw1_lag_3_mean']
                current_val = asset_df.iloc[i]['returnsClosePrevRaw1']
                
                if not pd.isna(lag_val):
                    assert not np.isclose(lag_val, current_val, rtol=1e-6), \
                        f"Look-ahead bias: lag equals current at index {i}"
    
    def test_imputation_uses_training_stats(self):
        """Test data should be imputed using training statistics."""
        train_data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=20),
            'assetCode': ['AAPL'] * 20,
            'returnsClosePrevRaw1': [0.01] * 20,
            'returnsOpenPrevRaw1': [0.01] * 20,
            'returnsClosePrevMktres1': [0.01] * 20,
            'returnsOpenPrevMktres1': [0.01] * 20,
            'volume': [1e6] * 20,
            'close': [100] * 20
        })
        
        test_data = pd.DataFrame({
            'time': pd.date_range('2024-01-21', periods=5),
            'assetCode': ['GOOGL'] * 5,
            'returnsClosePrevRaw1': [0.02] * 5,
            'returnsOpenPrevRaw1': [0.02] * 5,
            'returnsClosePrevMktres1': [0.02] * 5,
            'returnsOpenPrevMktres1': [0.02] * 5,
            'volume': [2e6] * 5,
            'close': [200] * 5
        })
        
        generator = LagFeatureGenerator(lag_windows=[3])
        generator.fit(train_data)
        
        train_median = generator.statistics_[
            'returnsClosePrevRaw1_lag_3_mean'
        ]['median']
        
        transformed = generator.transform(test_data)
        
        assert not transformed['returnsClosePrevRaw1_lag_3_mean'].isna().all()
    
    def test_handles_missing_values_in_features(self):
        """Should handle NaN in source features gracefully."""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=15),
            'assetCode': ['AAPL'] * 15,
            'returnsClosePrevRaw1': [
                0.01, np.nan, 0.02, np.nan, 0.03,
                0.01, 0.02, np.nan, 0.01, 0.02,
                0.01, 0.02, 0.03, 0.01, 0.02
            ],
            'returnsOpenPrevRaw1': [0.01] * 15,
            'returnsClosePrevMktres1': [0.01] * 15,
            'returnsOpenPrevMktres1': [0.01] * 15,
            'volume': [1e6] * 15,
            'close': [100] * 15
        })
        
        generator = LagFeatureGenerator(lag_windows=[3])
        generator.fit(data)
        transformed = generator.transform(data)
        
        lag_cols = [col for col in transformed.columns if 'lag' in col]
        assert not transformed[lag_cols].isna().all().any()
    
    def test_grouped_by_asset_independently(self):
        """Each asset should have independent lag calculations."""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10).tolist() * 2,
            'assetCode': ['AAPL'] * 10 + ['GOOGL'] * 10,
            'returnsClosePrevRaw1': list(range(10)) + list(range(100, 110)),
            'returnsOpenPrevRaw1': [0.01] * 20,
            'returnsClosePrevMktres1': [0.01] * 20,
            'returnsOpenPrevMktres1': [0.01] * 20,
            'volume': [1e6] * 20,
            'close': [100] * 20
        })
        
        generator = LagFeatureGenerator(lag_windows=[3])
        generator.fit(data)
        transformed = generator.transform(data)
        
        aapl_lags = transformed[
            transformed['assetCode'] == 'AAPL'
        ]['returnsClosePrevRaw1_lag_3_mean'].iloc[5]
        
        googl_lags = transformed[
            transformed['assetCode'] == 'GOOGL'
        ]['returnsClosePrevRaw1_lag_3_mean'].iloc[5]
        
        assert not np.isclose(aapl_lags, googl_lags, rtol=0.1)
    
    def test_error_on_missing_required_columns(self):
        """Should raise error if required columns missing."""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10),
            'assetCode': ['AAPL'] * 10
        })
        
        generator = LagFeatureGenerator()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.fit(data)
    
    def test_error_on_transform_before_fit(self):
        """Should raise error if transform called before fit."""
        generator = LagFeatureGenerator()
        data = pd.DataFrame({'assetCode': ['AAPL']})
        
        with pytest.raises(ValueError, match="Must call fit"):
            generator.transform(data)


class TestLeakageValidator:
    """Test look-ahead bias detection."""
    
    def test_detects_high_correlation(self):
        """Should detect suspiciously high correlation."""
        df = pd.DataFrame({
            'returnsClosePrevRaw1_lag_3_mean': [0.01, 0.02, -0.01, 0.03, 0.00],
            'returnsOpenNextMktres10': [0.01, 0.02, -0.01, 0.03, 0.00]
        })
        
        validator = LeakageValidator()
        result = validator.check_correlation_with_target(
            df,
            lag_columns=['returnsClosePrevRaw1_lag_3_mean'],
            threshold=0.9
        )
        
        assert result == False
    
    def test_allows_reasonable_correlation(self):
        """Should allow moderate correlation."""
        np.random.seed(42)
        
        target = np.random.randn(100) * 0.02
        lag = target + np.random.randn(100) * 0.03
        
        df = pd.DataFrame({
            'returnsClosePrevRaw1_lag_3_mean': lag,
            'returnsOpenNextMktres10': target
        })
        
        validator = LeakageValidator()
        result = validator.check_correlation_with_target(
            df,
            lag_columns=['returnsClosePrevRaw1_lag_3_mean'],
            threshold=0.9
        )
        
        assert result == True
    
    def test_verifies_temporal_ordering(self):
        """Should verify features are properly shifted."""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10),
            'assetCode': ['AAPL'] * 10,
            'returnsClosePrevRaw1': [0.01, 0.02, 0.03, 0.04, 0.05, 
                                     0.06, 0.07, 0.08, 0.09, 0.10]
        })
        
        # Manually create proper lag features
        # At index 3, lag_3_mean should be mean of [0.01, 0.02, 0.03] = 0.02
        data['returnsClosePrevRaw1_lag_3_mean'] = [
            np.nan, np.nan, np.nan,
            0.02,  # mean of [0.01, 0.02, 0.03]
            0.03,  # mean of [0.02, 0.03, 0.04]
            0.04,  # mean of [0.03, 0.04, 0.05]
            0.05,  # mean of [0.04, 0.05, 0.06]
            0.06,  # mean of [0.05, 0.06, 0.07]
            0.07,  # mean of [0.06, 0.07, 0.08]
            0.08   # mean of [0.07, 0.08, 0.09]
        ]
        
        validator = LeakageValidator()
        result = validator.verify_temporal_ordering(data)
        
        assert result == True
    
    def test_detects_incorrect_temporal_ordering(self):
        """Should detect when lag features use future data."""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10),
            'assetCode': ['AAPL'] * 10,
            'returnsClosePrevRaw1': [0.01, 0.02, 0.03, 0.04, 0.05, 
                                     0.06, 0.07, 0.08, 0.09, 0.10]
        })
        
        # Create INCORRECT lag features (using future data)
        # At index 3, incorrectly using mean of [0.02, 0.03, 0.04] instead of [0.01, 0.02, 0.03]
        data['returnsClosePrevRaw1_lag_3_mean'] = [
            np.nan, np.nan, np.nan,
            0.03,  # WRONG: should be 0.02
            0.04,  # WRONG: should be 0.03
            0.05,  # WRONG: should be 0.04
            0.06,  # WRONG: should be 0.05
            0.07,  # WRONG: should be 0.06
            0.08,  # WRONG: should be 0.07
            0.09   # WRONG: should be 0.08
        ]
        
        validator = LeakageValidator()
        result = validator.verify_temporal_ordering(data)
        
        assert result == False

class TestEdgeCaseHandler:
    """Test edge case handling."""
    
    def test_handles_new_assets(self):
        """Should impute new assets with training medians."""
        training_stats = {
            'volume_lag_3_mean': {
                'median': 1e6,
                'mean': 1.2e6,
                'std': 2e5
            }
        }
        
        df = pd.DataFrame({
            'assetCode': ['AAPL', 'GOOGL', 'NVDA'],
            'volume_lag_3_mean': [1.1e6, 1.2e6, np.nan]
        })
        
        handler = EdgeCaseHandler()
        result = handler.handle_new_assets(df, training_stats)
        
        assert not result.loc[result['assetCode'] == 'NVDA', 'volume_lag_3_mean'].isna().any()
        assert result.loc[result['assetCode'] == 'NVDA', 'volume_lag_3_mean'].iloc[0] == 1e6
    
    def test_detects_data_gaps(self):
        """Should detect significant gaps in time series."""
        dates = list(pd.date_range('2024-01-01', periods=10)) + \
                list(pd.date_range('2024-01-20', periods=5))
        
        df = pd.DataFrame({
            'time': dates,
            'assetCode': ['AAPL'] * 15,
            'volume': [1e6] * 15
        })
        
        handler = EdgeCaseHandler()
        result = handler.detect_data_gaps(df, max_gap_days=5)
        
        assert len(result) == 15
    
    def test_flags_insufficient_history(self):
        """Should flag assets with too few observations."""
        df = pd.DataFrame({
            'assetCode': ['AAPL'] * 30 + ['GOOGL'] * 2,
            'volume': [1e6] * 32
        })
        
        handler = EdgeCaseHandler()
        result = handler.flag_insufficient_history(df, min_observations=10)
        
        assert 'insufficient_history' in result.columns
        assert result[result['assetCode'] == 'GOOGL']['insufficient_history'].all()
        assert not result[result['assetCode'] == 'AAPL']['insufficient_history'].any()


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def realistic_data(self):
        """Market data with edge cases."""
        np.random.seed(42)
        data = []
        
        dates_aapl = pd.date_range('2024-01-01', periods=60)
        for date in dates_aapl:
            data.append({
                'time': date,
                'assetCode': 'AAPL',
                'assetName': 'Apple Inc',
                'returnsClosePrevRaw1': np.random.randn() * 0.02,
                'returnsOpenPrevRaw1': np.random.randn() * 0.02,
                'returnsClosePrevMktres1': np.random.randn() * 0.015,
                'returnsOpenPrevMktres1': np.random.randn() * 0.015,
                'volume': np.random.uniform(1e7, 5e7),
                'close': 150 + np.random.randn() * 10
            })
        
        dates_googl = list(pd.date_range('2024-01-01', periods=30)) + \
                      list(pd.date_range('2024-02-10', periods=20))
        for date in dates_googl:
            data.append({
                'time': date,
                'assetCode': 'GOOGL',
                'assetName': 'Alphabet Inc',
                'returnsClosePrevRaw1': np.random.randn() * 0.02,
                'returnsOpenPrevRaw1': np.random.randn() * 0.02,
                'returnsClosePrevMktres1': np.random.randn() * 0.015,
                'returnsOpenPrevMktres1': np.random.randn() * 0.015,
                'volume': np.random.uniform(1e7, 3e7),
                'close': 120 + np.random.randn() * 8
            })
        
        dates_nvda = pd.date_range('2024-02-15', periods=15)
        for date in dates_nvda:
            data.append({
                'time': date,
                'assetCode': 'NVDA',
                'assetName': 'NVIDIA Corp',
                'returnsClosePrevRaw1': np.random.randn() * 0.03,
                'returnsOpenPrevRaw1': np.random.randn() * 0.03,
                'returnsClosePrevMktres1': np.random.randn() * 0.025,
                'returnsOpenPrevMktres1': np.random.randn() * 0.025,
                'volume': np.random.uniform(5e6, 2e7),
                'close': 500 + np.random.randn() * 30
            })
        
        return pd.DataFrame(data)
    
    def test_full_pipeline(self, realistic_data):
        """Test complete workflow with edge cases."""
        df = realistic_data
        
        train = df[df['time'] < '2024-02-01'].copy()
        test = df[df['time'] >= '2024-02-01'].copy()
        
        generator = LagFeatureGenerator(
            lag_windows=[3, 7, 14],
            feature_columns=[
                'returnsClosePrevRaw1',
                'returnsOpenPrevRaw1',
                'volume'
            ]
        )
        
        generator.fit(train)
        
        train_transformed = generator.transform(train)
        test_transformed = generator.transform(test)
        
        validator = LeakageValidator()
        lag_cols = [col for col in train_transformed.columns if 'lag' in col]
        
        assert len(lag_cols) > 0
        assert not train_transformed[lag_cols].isna().all().any()
        assert not test_transformed[lag_cols].isna().all().any()
        
        assert 'NVDA' in test_transformed['assetCode'].values
        nvda_data = test_transformed[test_transformed['assetCode'] == 'NVDA']
        assert not nvda_data[lag_cols[:5]].isna().all().any()
        
        assert train_transformed['time'].max() < test_transformed['time'].min()
        
        assert validator.verify_temporal_ordering(train_transformed)
