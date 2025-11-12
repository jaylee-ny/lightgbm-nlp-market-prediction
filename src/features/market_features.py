from typing import List
import pandas as pd
import numpy as np
from loguru import logger


class MarketFeatureEngineer:
    """Engineer market microstructure features."""
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize market feature engineer.
        
        Args:
            window_sizes: Rolling window sizes for volatility calculations
        """
        self.window_sizes = window_sizes or [5, 10, 20]
        logger.info(f"MarketFeatureEngineer initialized with windows={self.window_sizes}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all market features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        df = self._compute_returns(df)
        df = self._compute_volatility(df)
        df = self._compute_vwap(df)
        df = self._compute_liquidity_metrics(df)
        df = self._compute_microstructure_features(df)
        
        return df
    
    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics."""
        if 'close' in df.columns:
            df['simple_return'] = df.groupby('asset_code')['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        if 'open' in df.columns and 'close' in df.columns:
            df['intraday_return'] = (df['close'] - df['open']) / df['open']
            df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility estimators."""
        for window in self.window_sizes:
            if 'simple_return' in df.columns:
                df[f'volatility_{window}d'] = (
                    df.groupby('asset_code')['simple_return']
                    .rolling(window).std()
                    .reset_index(0, drop=True)
                )
            
            if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
                df[f'parkinson_volatility_{window}d'] = (
                    self._parkinson_volatility(df, window)
                )
                df[f'garman_klass_volatility_{window}d'] = (
                    self._garman_klass_volatility(df, window)
                )
        
        return df
    
    def _parkinson_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        Parkinson volatility estimator (uses high-low range).
        More efficient than close-to-close.
        """
        hl_ratio = np.log(df['high'] / df['low'])
        parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
        
        return (
            df.groupby('asset_code')['asset_code']
            .transform(lambda x: parkinson_var.loc[x.index].rolling(window).mean())
            .apply(np.sqrt)
        )
    
    def _garman_klass_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        Garman-Klass volatility estimator (uses OHLC).
        More efficient than Parkinson.
        """
        hl = np.log(df['high'] / df['low']) ** 2
        co = np.log(df['close'] / df['open']) ** 2
        
        gk_var = 0.5 * hl - (2 * np.log(2) - 1) * co
        
        return (
            df.groupby('asset_code')['asset_code']
            .transform(lambda x: gk_var.loc[x.index].rolling(window).mean())
            .apply(np.sqrt)
        )
    
    def _compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-weighted average price."""
        if all(col in df.columns for col in ['close', 'volume']):
            df['vwap'] = df['close']
            
            for window in self.window_sizes:
                vwap_col = f'vwap_{window}d'
                df[vwap_col] = np.nan
                
                for asset in df['asset_code'].unique():
                    mask = df['asset_code'] == asset
                    asset_data = df.loc[mask]
                    
                    price_volume = asset_data['close'] * asset_data['volume']
                    numerator = price_volume.rolling(window).sum()
                    denominator = asset_data['volume'].rolling(window).sum()
                    
                    df.loc[mask, vwap_col] = numerator / denominator
        
        return df
    
    def _compute_liquidity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity metrics."""
        if 'volume' in df.columns:
            for window in self.window_sizes:
                df[f'volume_avg_{window}d'] = (
                    df.groupby('asset_code')['volume']
                    .rolling(window).mean()
                    .reset_index(0, drop=True)
                )
                
                df[f'volume_std_{window}d'] = (
                    df.groupby('asset_code')['volume']
                    .rolling(window).std()
                    .reset_index(0, drop=True)
                )
        
        if 'close' in df.columns and 'volume' in df.columns:
            df['dollar_volume'] = df['close'] * df['volume']
            
            for window in self.window_sizes:
                df[f'dollar_volume_avg_{window}d'] = (
                    df.groupby('asset_code')['dollar_volume']
                    .rolling(window).mean()
                    .reset_index(0, drop=True)
                )
        
        return df
    
    def _compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure indicators."""
        if 'simple_return' in df.columns and 'volume' in df.columns:
            for window in self.window_sizes:
                corr_df = (
                    df.groupby('asset_code')[['simple_return', 'volume']]
                    .rolling(window)
                    .corr()
                )
                corr_series = (
                    corr_df.unstack()['simple_return']['volume']
                )
                df[f'price_volume_corr_{window}d'] = corr_series.reset_index(level=0, drop=True)
                

        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        return df
