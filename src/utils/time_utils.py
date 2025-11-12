from typing import Iterator, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class TimeSeriesSplitter:
    """
    Walk-forward time-series cross-validation splitter.
    
    Ensures no look-ahead bias by splitting data with strict temporal ordering.
    Train period is always before test period with no overlap.
    """
    
    def __init__(
        self,
        train_window_years: int = 3,
        test_window_months: int = 6,
        gap_days: int = 0
    ):
        """
        Initialize time-series splitter.
        
        Args:
            train_window_years: Size of training window in years
            test_window_months: Size of test window in months
            gap_days: Gap between train and test to account for data availability delays
        """
        self.train_window_years = train_window_years
        self.test_window_months = test_window_months
        self.gap_days = gap_days
        
        logger.info(
            f"TimeSeriesSplitter initialized: "
            f"train_window={train_window_years}y, "
            f"test_window={test_window_months}m, "
            f"gap={gap_days}d"
        )
    
    def split(
        self,
        df: pd.DataFrame,
        date_column: str = 'time'
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits with walk-forward validation.
        
        Args:
            df: DataFrame with datetime column
            date_column: Name of datetime column
            
        Yields:
            Tuple of (train_df, test_df) for each fold
            
        Raises:
            ValueError: If date column missing or insufficient data
        """
        if date_column not in df.columns:
            raise ValueError(f"DataFrame missing '{date_column}' column")
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            raise ValueError(f"Column '{date_column}' must be datetime type")
        
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        
        min_date = df_sorted[date_column].min()
        max_date = df_sorted[date_column].max()
        
        logger.info(f"Data range: {min_date} to {max_date}")
        
        train_delta = timedelta(days=365 * self.train_window_years)
        test_delta = timedelta(days=30 * self.test_window_months)
        gap_delta = timedelta(days=self.gap_days)
        
        current_train_end = min_date + train_delta
        
        fold = 0
        while True:
            train_start = current_train_end - train_delta
            train_end = current_train_end
            test_start = train_end + gap_delta
            test_end = test_start + test_delta
            
            if test_end > max_date:
                logger.info(f"Reached end of data after {fold} folds")
                break
            
            train_mask = (
                (df_sorted[date_column] >= train_start) &
                (df_sorted[date_column] < train_end)
            )
            test_mask = (
                (df_sorted[date_column] >= test_start) &
                (df_sorted[date_column] < test_end)
            )
            
            train_df = df_sorted[train_mask].copy()
            test_df = df_sorted[test_mask].copy()
            
            if len(train_df) == 0:
                logger.warning(f"Fold {fold}: No training data, skipping")
                current_train_end += test_delta
                continue
            
            if len(test_df) == 0:
                logger.warning(f"Fold {fold}: No test data, skipping")
                current_train_end += test_delta
                continue
            
            self.validate_no_leakage(train_df, test_df, date_column)
            
            fold += 1
            logger.info(
                f"Fold {fold}: "
                f"train=[{train_start.date()}, {train_end.date()}), "
                f"test=[{test_start.date()}, {test_end.date()}), "
                f"train_size={len(train_df)}, test_size={len(test_df)}"
            )
            
            yield train_df, test_df
            
            current_train_end += test_delta
    
    def validate_no_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_column: str = 'time'
    ) -> bool:
        """
        Validate that train period is strictly before test period.
        
        Args:
            train_df: Training data
            test_df: Test data
            date_column: Name of datetime column
            
        Returns:
            True if no leakage detected
            
        Raises:
            ValueError: If temporal leakage detected
        """
        train_max = train_df[date_column].max()
        test_min = test_df[date_column].min()
        
        if train_max >= test_min:
            raise ValueError(
                f"Temporal leakage detected! "
                f"Train max date ({train_max}) >= Test min date ({test_min})"
            )
        
        return True
    
    def get_n_splits(self, df: pd.DataFrame, date_column: str = 'time') -> int:
        """
        Calculate number of splits that will be generated.
        
        Args:
            df: DataFrame with datetime column
            date_column: Name of datetime column
            
        Returns:
            Number of folds
        """
        splits = list(self.split(df, date_column))
        return len(splits)


class ExpandingWindowSplitter:
    """
    Expanding window cross-validation splitter.
    
    Train window grows with each fold, test window is fixed.
    Useful when you want to use all available historical data.
    """
    
    def __init__(
        self,
        initial_train_years: int = 3,
        test_window_months: int = 6,
        gap_days: int = 0
    ):
        """
        Initialize expanding window splitter.
        
        Args:
            initial_train_years: Initial training window size in years
            test_window_months: Size of test window in months
            gap_days: Gap between train and test
        """
        self.initial_train_years = initial_train_years
        self.test_window_months = test_window_months
        self.gap_days = gap_days
        
        logger.info(
            f"ExpandingWindowSplitter initialized: "
            f"initial_train={initial_train_years}y, "
            f"test_window={test_window_months}m"
        )
    
    def split(
        self,
        df: pd.DataFrame,
        date_column: str = 'time'
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits with expanding window.
        
        Args:
            df: DataFrame with datetime column
            date_column: Name of datetime column
            
        Yields:
            Tuple of (train_df, test_df) for each fold
        """
        if date_column not in df.columns:
            raise ValueError(f"DataFrame missing '{date_column}' column")
        
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        
        min_date = df_sorted[date_column].min()
        max_date = df_sorted[date_column].max()
        
        test_delta = timedelta(days=30 * self.test_window_months)
        gap_delta = timedelta(days=self.gap_days)
        initial_train_delta = timedelta(days=365 * self.initial_train_years)
        
        train_start = min_date
        current_train_end = min_date + initial_train_delta
        
        fold = 0
        while True:
            train_end = current_train_end
            test_start = train_end + gap_delta
            test_end = test_start + test_delta
            
            if test_end > max_date:
                logger.info(f"Reached end of data after {fold} folds")
                break
            
            train_mask = (
                (df_sorted[date_column] >= train_start) &
                (df_sorted[date_column] < train_end)
            )
            test_mask = (
                (df_sorted[date_column] >= test_start) &
                (df_sorted[date_column] < test_end)
            )
            
            train_df = df_sorted[train_mask].copy()
            test_df = df_sorted[test_mask].copy()
            
            if len(train_df) == 0 or len(test_df) == 0:
                current_train_end += test_delta
                continue
            
            fold += 1
            logger.info(
                f"Fold {fold}: "
                f"train=[{train_start.date()}, {train_end.date()}), "
                f"test=[{test_start.date()}, {test_end.date()}), "
                f"train_size={len(train_df)}, test_size={len(test_df)}"
            )
            
            yield train_df, test_df
            
            current_train_end += test_delta


def convert_date_to_int(date: datetime) -> int:
    """
    Convert datetime to integer in YYYYMMDD format.
    
    Args:
        date: Datetime object
        
    Returns:
        Date as integer (e.g., 20240315)
    """
    return 10000 * date.year + 100 * date.month + date.day


def convert_int_to_date(date_int: int) -> datetime:
    """
    Convert integer date to datetime.
    
    Args:
        date_int: Date as integer in YYYYMMDD format
        
    Returns:
        Datetime object
    """
    return datetime.strptime(str(date_int), '%Y%m%d')


def get_trading_days(
    start_date: datetime,
    end_date: datetime,
    df: pd.DataFrame,
    date_column: str = 'time'
) -> np.ndarray:
    """
    Extract unique trading days from data.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        df: DataFrame with trading data
        date_column: Name of datetime column
        
    Returns:
        Sorted array of trading days as integers (YYYYMMDD)
    """
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    dates = df.loc[mask, date_column].unique()
    dates_sorted = np.sort(dates)
    
    trading_days = np.array([
        convert_date_to_int(pd.Timestamp(d).to_pydatetime())
        for d in dates_sorted
    ])
    
    return trading_days


def map_to_next_trading_day(
    date: int,
    trading_days: np.ndarray
) -> int:
    """
    Map a date to the next available trading day.
    
    Args:
        date: Date as integer (YYYYMMDD)
        trading_days: Array of trading days as integers
        
    Returns:
        Next trading day as integer
        
    Raises:
        ValueError: If date is after all trading days
    """
    if date in trading_days:
        return date
    
    future_days = trading_days[trading_days > date]
    
    if len(future_days) == 0:
        raise ValueError(
            f"Date {date} is after all available trading days. "
            f"Latest trading day: {trading_days[-1]}"
        )
    
    return future_days[0]
