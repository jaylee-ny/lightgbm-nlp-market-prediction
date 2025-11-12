from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin


class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate lag features with fit/transform pattern.
    
    Prevents look-ahead bias by:
    1. Learning imputation statistics only from training data
    2. Using shift() to ensure features use only past information
    3. Computing rolling statistics with proper temporal ordering
    
    Parameters
    ----------
    lag_windows : List[int]
        Rolling window sizes in days (e.g., [3, 7, 14, 30])
    feature_columns : List[str]
        Columns to create lag features for
    group_column : str
        Column to group by, typically 'assetCode'
    min_periods : int
        Minimum observations required for rolling statistics
    shift_size : int
        Number of periods to shift (1 = use previous day)
    """
    
    def __init__(
        self,
        lag_windows: List[int] = None,
        feature_columns: List[str] = None,
        group_column: str = 'assetCode',
        min_periods: int = 1,
        shift_size: int = 1
    ):
        self.lag_windows = lag_windows or [3, 7, 14, 30]
        self.feature_columns = feature_columns or [
            'returnsClosePrevRaw1',
            'returnsOpenPrevRaw1',
            'returnsClosePrevMktres1',
            'returnsOpenPrevMktres1',
            'volume',
            'close'
        ]
        self.group_column = group_column
        self.min_periods = min_periods
        self.shift_size = shift_size
        
        self.statistics_ = {}
        self.feature_names_ = []
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn imputation statistics from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data with time series features
        y : ignored
            Not used, present for sklearn compatibility
            
        Returns
        -------
        self : LagFeatureGenerator
        """
        logger.info("Fitting LagFeatureGenerator...")
        self._validate_input(X)
        
        X_with_lags = self._generate_lag_features(X.copy())
        
        lag_columns = [
            col for col in X_with_lags.columns 
            if any(f'_lag_{w}_' in col for w in self.lag_windows)
        ]
        
        for col in lag_columns:
            valid_data = X_with_lags[col].dropna()
            if len(valid_data) > 0:
                self.statistics_[col] = {
                    'mean': valid_data.mean(),
                    'median': valid_data.median(),
                    'std': valid_data.std(),
                    'min': valid_data.min(),
                    'max': valid_data.max()
                }
            else:
                self.statistics_[col] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        
        self.feature_names_ = lag_columns
        self._is_fitted = True
        
        logger.info(f"Fitted with {len(self.feature_names_)} lag features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features to data using fitted statistics for imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Data with lag features added
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        logger.debug(f"Transforming {len(X)} observations...")
        
        X_transformed = self._generate_lag_features(X.copy())
        
        for col in self.feature_names_:
            if col in X_transformed.columns:
                fill_value = self.statistics_[col]['median']
                X_transformed[col].fillna(fill_value, inplace=True)
            else:
                logger.warning(f"Expected feature {col} not found, creating with median")
                X_transformed[col] = self.statistics_[col]['median']
        
        return X_transformed
    
    def _generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lag features grouped by asset.
        
        Uses shift() to ensure no future information leaks into features.
        """
        if self.group_column not in df.columns:
            raise ValueError(f"Group column '{self.group_column}' not found")
        
        if 'time' in df.columns:
            df = df.sort_values([self.group_column, 'time'])
        else:
            logger.warning("No 'time' column found, assuming data is sorted")
        
        for feature in self.feature_columns:
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' not found, skipping")
                continue
            
            for window in self.lag_windows:
                shifted = df.groupby(self.group_column)[feature].shift(self.shift_size)
                
                rolled = shifted.rolling(
                    window=window,
                    min_periods=self.min_periods
                )
                
                df[f'{feature}_lag_{window}_mean'] = rolled.mean()
                df[f'{feature}_lag_{window}_std'] = rolled.std()
                df[f'{feature}_lag_{window}_max'] = rolled.max()
                df[f'{feature}_lag_{window}_min'] = rolled.min()
                
                if window >= 7:
                    df[f'{feature}_lag_{window}_ratio'] = (
                        df[feature] / (rolled.mean() + 1e-8)
                    )
        
        return df
    
    def _validate_input(self, X: pd.DataFrame):
        """Validate required columns exist."""
        required_cols = [self.group_column] + self.feature_columns
        missing = set(required_cols) - set(X.columns)
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(X) == 0:
            raise ValueError("Empty DataFrame provided")
    
    def get_feature_names_out(self):
        """Return names of generated features."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before get_feature_names_out()")
        return self.feature_names_


class LeakageValidator:
    """Validate lag features don't contain look-ahead bias."""
    
    @staticmethod
    def check_correlation_with_target(
        df: pd.DataFrame,
        lag_columns: List[str],
        target_column: str = 'returnsOpenNextMktres10',
        threshold: float = 0.9
    ) -> bool:
        """
        Check if lag features correlate suspiciously high with target.
        
        High correlation (>0.9) suggests possible look-ahead bias.
        """
        for lag_col in lag_columns:
            if lag_col in df.columns and target_column in df.columns:
                clean_data = df[[lag_col, target_column]].dropna()
                if len(clean_data) > 0:
                    corr = clean_data.corr().iloc[0, 1]
                    
                    if abs(corr) > threshold:
                        logger.error(
                            f"Suspicious correlation detected: {lag_col} has "
                            f"{corr:.3f} correlation with {target_column}"
                        )
                        return False
        
        return True
    
    @staticmethod
    def verify_temporal_ordering(
        df: pd.DataFrame,
        group_col: str = 'assetCode',
        sample_size: int = 5
    ) -> bool:
        """
        Verify lag features are computed using only past data.
        
        Checks that for a 3-day lag mean, the value at time T
        is the mean of values at T-1, T-2, T-3, not T, T-1, T-2.
        """
        lag_cols = [col for col in df.columns if '_lag_' in col and '_mean' in col]
        
        if not lag_cols:
            logger.warning("No lag features found to validate")
            return True
        
        for lag_col in lag_cols[:3]:  # Check first 3 lag features
            parts = lag_col.split('_lag_')
            if len(parts) != 2:
                continue
            
            base_feature = parts[0]
            window_and_stat = parts[1].split('_')
            
            if len(window_and_stat) < 2:
                continue
            
            try:
                window = int(window_and_stat[0])
            except ValueError:
                continue
            
            if base_feature not in df.columns:
                continue
            
            for asset in df[group_col].unique()[:sample_size]:
                asset_df = df[df[group_col] == asset].sort_values('time').reset_index(drop=True)
                
                if len(asset_df) < window + 5:
                    continue
                
                for i in range(window + 2, min(len(asset_df), window + 7)):
                    lag_val = asset_df.loc[i, lag_col]
                    
                    if pd.isna(lag_val):
                        continue
                    
                    past_values = asset_df.loc[i-window:i-1, base_feature]
                    
                    if len(past_values) == window and not past_values.isna().any():
                        expected_mean = past_values.mean()
                        
                        if not np.isclose(lag_val, expected_mean, rtol=0.01):
                            logger.error(
                                f"Temporal ordering issue in {asset}: "
                                f"lag feature {lag_val:.6f} != expected {expected_mean:.6f}"
                            )
                            return False
        
        return True

class EdgeCaseHandler:
    """Handle edge cases in lag feature generation."""
    
    @staticmethod
    def handle_new_assets(
        df: pd.DataFrame,
        training_stats: Dict,
        group_col: str = 'assetCode'
    ) -> pd.DataFrame:
        """
        Handle assets appearing in test but not training (IPOs).
        
        Uses cross-asset medians from training data for imputation.
        """
        if not training_stats:
            return df
        
        test_assets = set(df[group_col].unique())
        known_features = set(training_stats.keys())
        
        global_medians = {
            feature: stats['median'] 
            for feature, stats in training_stats.items()
        }
        
        for asset in test_assets:
            asset_data = df[df[group_col] == asset]
            
            for feature in known_features:
                if feature in df.columns:
                    mask = (df[group_col] == asset) & df[feature].isna()
                    df.loc[mask, feature] = global_medians[feature]
        
        return df
    
    @staticmethod
    def detect_data_gaps(
        df: pd.DataFrame,
        group_col: str = 'assetCode',
        time_col: str = 'time',
        max_gap_days: int = 7
    ) -> pd.DataFrame:
        """
        Detect significant gaps in time series data.
        
        Gaps may indicate delistings, trading halts, or data quality issues.
        """
        gaps_found = []
        
        for asset in df[group_col].unique():
            asset_df = df[df[group_col] == asset].sort_values(time_col)
            
            if len(asset_df) < 2:
                continue
            
            if pd.api.types.is_datetime64_any_dtype(asset_df[time_col]):
                time_diffs = asset_df[time_col].diff()
                large_gaps = time_diffs > pd.Timedelta(days=max_gap_days)
            else:
                time_diffs = asset_df[time_col].diff()
                large_gaps = time_diffs > max_gap_days
            
            if large_gaps.any():
                gaps_found.append({
                    'asset': asset,
                    'n_gaps': large_gaps.sum(),
                    'max_gap': time_diffs.max()
                })
        
        if gaps_found:
            logger.warning(
                f"Found data gaps in {len(gaps_found)} assets. "
                f"Lag features near gaps may be unreliable."
            )
        
        return df
    
    @staticmethod
    def flag_insufficient_history(
        df: pd.DataFrame,
        min_observations: int,
        group_col: str = 'assetCode'
    ) -> pd.DataFrame:
        """
        Flag assets with insufficient history for longest lag window.
        """
        asset_counts = df[group_col].value_counts()
        insufficient = asset_counts[asset_counts < min_observations]
        
        if len(insufficient) > 0:
            logger.info(
                f"{len(insufficient)} assets have < {min_observations} observations"
            )
            df['insufficient_history'] = df[group_col].isin(insufficient.index)
        else:
            df['insufficient_history'] = False
        
        return df
