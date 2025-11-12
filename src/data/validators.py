from typing import List, Dict
import pandas as pd
import numpy as np
from loguru import logger


class DataValidator:
    """Validate data quality and schema."""
    
    def validate_no_missing_required_columns(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str]
    ) -> bool:
        """
        Check that all required columns are present.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all columns present
            
        Raises:
            ValueError: If columns are missing
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    def validate_no_duplicate_timestamps(
        self, 
        df: pd.DataFrame,
        groupby_columns: List[str] = None
    ) -> bool:
        """
        Check for duplicate timestamps within groups.
        
        Args:
            df: DataFrame to validate
            groupby_columns: Columns to group by (e.g., ['time', 'assetCode'])
            
        Returns:
            True if no duplicates
            
        Raises:
            ValueError: If duplicates found
        """
        if groupby_columns is None:
            groupby_columns = ['time', 'assetCode']
        
        duplicates = df.duplicated(subset=groupby_columns, keep=False)
        
        if duplicates.any():
            duplicate_count = duplicates.sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamp-asset pairs")
            raise ValueError(f"Found {duplicate_count} duplicate entries")
        
        return True
    
    def validate_price_sanity(self, df: pd.DataFrame) -> bool:
        """
        Check that prices are within reasonable ranges.
        
        Args:
            df: DataFrame with 'open' and 'close' columns
            
        Returns:
            True if prices are valid
            
        Raises:
            ValueError: If invalid prices found
        """
        if 'close' in df.columns:
            negative_prices = df['close'] < 0
            if negative_prices.any():
                count = negative_prices.sum()
                raise ValueError(f"Found {count} negative closing prices")
            
            zero_prices = df['close'] == 0
            if zero_prices.any():
                count = zero_prices.sum()
                logger.warning(f"Found {count} zero closing prices")
        
        if 'open' in df.columns:
            negative_prices = df['open'] < 0
            if negative_prices.any():
                count = negative_prices.sum()
                raise ValueError(f"Found {count} negative opening prices")
        
        return True
    
    def validate_temporal_ordering(self, df: pd.DataFrame) -> bool:
        """
        Check that timestamps are in chronological order within groups.
        
        Args:
            df: DataFrame with 'time' column
            
        Returns:
            True if properly ordered
            
        Raises:
            ValueError: If ordering is incorrect
        """
        if 'time' not in df.columns:
            raise ValueError("DataFrame must contain 'time' column")
        
        if 'assetCode' in df.columns:
            for asset in df['assetCode'].unique():
                asset_df = df[df['assetCode'] == asset]
                if not asset_df['time'].is_monotonic_increasing:
                    raise ValueError(f"Time not chronological for asset {asset}")
        else:
            if not df['time'].is_monotonic_increasing:
                raise ValueError("Time not in chronological order")
        
        return True
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.to_dict(),
        }
        
        if 'time' in df.columns:
            report['date_range'] = {
                'min': df['time'].min(),
                'max': df['time'].max(),
                'days': (df['time'].max() - df['time'].min()).days
            }
        
        if 'assetCode' in df.columns:
            report['unique_assets'] = df['assetCode'].nunique()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            report['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        logger.info(f"Data quality report generated for {len(df)} rows")
        
        return report
